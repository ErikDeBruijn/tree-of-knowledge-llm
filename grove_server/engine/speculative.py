"""Self-speculative decoding using layer-skip draft model.

Uses the same base model with two different skip_layers configurations:
- Draft: aggressive skip (many layers skipped) for cheap token generation
- Verify: normal skip (few layers skipped) for accurate verification

The draft generates K candidate tokens autoregressively, then the verify
model checks them all in one batched forward pass. Tokens are accepted
up to the first mismatch, plus a bonus token from the verify model.

Both models share the same FP8 weight storage (no duplicate VRAM).
"""

from __future__ import annotations

import torch

from grove_server.engine.graphable_model import FP8GraphableDecodeStep
from grove_server.engine.static_kv_cache import StaticKVCache

HAS_CUDA = torch.cuda.is_available()


class SelfSpeculativeDecoder:
    """Self-speculative decoding using layer-skip draft model.

    Draft: FP8GraphableDecodeStep with many layers skipped (cheap)
    Verify: FP8GraphableDecodeStep with few layers skipped (accurate)

    Both share the same FP8 weight storage — no duplicate VRAM cost.

    Optionally captures the draft decode step as a CUDA graph for
    zero-overhead kernel launch during the K-step draft loop.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        draft_skip_layers: list[int],
        verify_skip_layers: list[int],
        max_seq_len: int = 2048,
        draft_tokens: int = 6,
    ) -> None:
        self.draft_tokens = draft_tokens

        # Draft graph state (populated by capture_draft_graph)
        self._draft_graph: torch.cuda.CUDAGraph | None = None  # type: ignore[attr-defined]
        self._draft_eager_fn = None  # CPU fallback closure
        self._draft_static_tok: torch.Tensor | None = None
        self._draft_static_pos: torch.Tensor | None = None
        self._draft_static_logits: torch.Tensor | None = None

        config = model.config
        num_layers = config.num_hidden_layers
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim
        dtype = next(model.parameters()).dtype
        device = str(next(model.parameters()).device)

        # Create separate KV caches for draft and verify
        draft_cache = StaticKVCache(
            num_layers=num_layers, num_heads=num_kv_heads,
            head_dim=head_dim, max_seq_len=max_seq_len,
            batch_size=1, dtype=dtype, device=device,
        )
        verify_cache = StaticKVCache(
            num_layers=num_layers, num_heads=num_kv_heads,
            head_dim=head_dim, max_seq_len=max_seq_len,
            batch_size=1, dtype=dtype, device=device,
        )

        # Create verify model with NO skip_layers first so it quantizes ALL weights.
        # Both draft and verify need access to different subsets of layers,
        # so we must quantize everything before setting skip configs.
        self.verify = FP8GraphableDecodeStep(
            model, verify_cache, max_seq_len,
            skip_layers=[],  # Quantize everything
        )
        # Now set the actual skip config
        self.verify.skip_layers = set(verify_skip_layers or [])

        # Create draft model sharing verify's FP8 weights
        # Skip _precompute_fp8_weights since weights are already freed
        self.draft = _make_shared_fp8_step(
            model, draft_cache, max_seq_len,
            skip_layers=draft_skip_layers,
            shared_fp8_weights=self.verify.fp8_weights,
            use_scaled_mm=self.verify._use_scaled_mm,
            x_scale=self.verify._x_scale,
        )

    @property
    def draft_graph_captured(self) -> bool:
        """Whether the draft decode step has been captured as a CUDA graph."""
        return self._draft_graph is not None or self._draft_eager_fn is not None

    def capture_draft_graph(self) -> None:
        """Capture the full K-step draft loop as a single CUDA graph.

        The draft cache is always reset and refilled at position 0 before
        drafting, so positions 1..K are deterministic. This lets us capture
        all K decode steps as one graph.

        Prerequisites: draft cache must be at seq_len=1 (one token prefilled).

        On CPU, stores a closure for eager replay with the same API.
        """
        device = str(next(self.draft.parameters()).device)
        use_cuda = HAS_CUDA and "cuda" in device

        K = self.draft_tokens

        # Static input token buffer -- start token is copied here before replay
        self._draft_static_tok = torch.zeros(1, 1, dtype=torch.long, device=device)

        # Pre-allocate position tensors (cannot allocate inside graph capture)
        self._draft_pos_tensors = [
            torch.tensor([[step + 1]], device=device) for step in range(K)
        ]

        # Static output buffers: one logits tensor per draft step
        self._draft_static_logits_list: list[torch.Tensor] = []

        if use_cuda:
            # Ensure draft cache is at position 1 (after prefill)
            assert self.draft.cache.seq_len == 1, (
                f"Draft cache must be at seq_len=1 before capture, got {self.draft.cache.seq_len}"
            )

            # Warmup: run the full K-step loop 3 times to stabilize allocations
            for _ in range(3):
                self.draft.cache.seq_len = 1  # reset to post-prefill
                tok = self._draft_static_tok
                for step in range(K):
                    logits = self.draft(tok, self._draft_pos_tensors[step])
                    tok = logits[:, -1:].argmax(dim=-1)
            torch.cuda.synchronize()

            # Reset to capture position
            self.draft.cache.seq_len = 1

            # Capture the full K-step loop
            self._draft_graph = torch.cuda.CUDAGraph()
            self._draft_static_logits_list = []
            with torch.cuda.graph(self._draft_graph):
                tok = self._draft_static_tok
                for step in range(K):
                    logits = self.draft(tok, self._draft_pos_tensors[step])
                    self._draft_static_logits_list.append(logits)
                    tok = logits[:, -1:].argmax(dim=-1)

            # After capture, cache.seq_len was advanced K times by Python code.
            # Reset it -- we'll manage it manually in draft_k_tokens_graphed.
            # (The graph baked in the correct per-step positions.)
        else:
            # CPU fallback: store enough state for eager replay
            self._draft_graph = None
            self._draft_eager_fn = self._eager_draft_k

    def _eager_draft_k(self) -> list[torch.Tensor]:
        """Eager K-step draft loop for CPU fallback."""
        device = self._draft_static_tok.device
        tokens = []
        tok = self._draft_static_tok
        for step in range(self.draft_tokens):
            pos = torch.tensor([[step + 1]], device=device)
            logits = self.draft(tok, pos)
            next_tok = logits[:, -1:].argmax(dim=-1)
            tokens.append(next_tok.squeeze())
            tok = next_tok
        return tokens

    def draft_k_tokens_graphed(
        self,
        start_token: torch.Tensor,
        start_pos: int,
    ) -> list[torch.Tensor]:
        """Generate K draft tokens using CUDA graph replay (or eager fallback).

        The draft cache must be at seq_len=1 (position 0 prefilled).
        start_pos must be 1 (the graph was captured with positions 1..K).

        Args:
            start_token: First token to feed into draft, shape (1, 1).
            start_pos: Position ID for the first draft token (must be 1).

        Returns:
            List of K token tensors (each scalar).
        """
        self._draft_static_tok.copy_(start_token)

        if self._draft_graph is not None:
            # Reset cache to post-prefill state (graph expects seq_len=1)
            self.draft.cache.seq_len = 1
            self._draft_graph.replay()
            # Extract tokens from static logits
            tokens = []
            for logits in self._draft_static_logits_list:
                next_tok = logits[:, -1:].argmax(dim=-1)
                tokens.append(next_tok.squeeze())
            # Set cache to final state
            self.draft.cache.seq_len = 1 + self.draft_tokens
            return tokens
        else:
            # CPU eager fallback
            self.draft.cache.seq_len = 1
            return self._eager_draft_k()

    def draft_k_tokens(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate K draft tokens autoregressively using the draft model.

        Args:
            input_ids: Starting token (1, 1).
            position_ids: Starting position (1, 1).

        Returns:
            Tuple of (draft_token_ids [K], draft_logits [K, vocab]).
        """
        draft_token_ids = []
        draft_logits_list = []

        current_id = input_ids
        current_pos = position_ids

        for _ in range(self.draft_tokens):
            logits = self.draft(current_id, current_pos)  # (1, 1, V)
            logits_2d = logits.squeeze(0).squeeze(0)  # (V,)
            next_token = logits_2d.argmax().item()

            draft_token_ids.append(next_token)
            draft_logits_list.append(logits_2d)

            current_id = torch.tensor([[next_token]], device=input_ids.device)
            current_pos = current_pos + 1

        tokens = torch.tensor(draft_token_ids, device=input_ids.device)
        all_logits = torch.stack(draft_logits_list)  # (K, V)
        return tokens, all_logits

    def speculative_step(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """One speculative decoding step.

        1. Process the input token through verify model (build KV cache)
        2. Draft K tokens using draft model (cheap)
        3. Verify all K tokens in one batched forward pass
        4. Accept matching prefix + correction token

        Args:
            input_ids: Current token (1, 1).
            position_ids: Current position (1, 1).

        Returns:
            Tuple of (accepted_token_ids [n], n_accepted).
        """
        device = input_ids.device

        # Step 1: Process input token through verify model to build its KV cache
        verify_logits_0 = self.verify(input_ids, position_ids)  # (1, 1, V)
        first_verify_token = verify_logits_0.squeeze(0).squeeze(0).argmax().item()

        # Step 2: Also process through draft model, then generate K draft tokens
        self.draft.cache.reset()
        _ = self.draft(input_ids, position_ids)  # Build draft KV cache
        draft_start_pos = position_ids.item() + 1

        # Feed the verify model's predicted token as the first draft input
        draft_input = torch.tensor([[first_verify_token]], device=device)

        if self.draft_graph_captured:
            # Use CUDA graph replay for the draft loop
            draft_tok_tensors = self.draft_k_tokens_graphed(
                draft_input, start_pos=draft_start_pos,
            )
            draft_token_ids = [t.item() for t in draft_tok_tensors]
        else:
            # Eager draft loop
            draft_pos = position_ids + 1
            draft_token_ids = []
            for _ in range(self.draft_tokens):
                logits = self.draft(draft_input, draft_pos)
                next_token = logits.squeeze(0).squeeze(0).argmax().item()
                draft_token_ids.append(next_token)
                draft_input = torch.tensor([[next_token]], device=device)
                draft_pos = draft_pos + 1

        if not draft_token_ids:
            self.draft.cache.reset()
            return torch.tensor([first_verify_token], device=device), 1

        draft_tokens = torch.tensor(draft_token_ids, device=device)

        # Step 3: Verify all draft tokens in one batched forward pass
        # Feed all draft tokens at once to verify model
        verify_input = draft_tokens.unsqueeze(0)  # (1, K)
        verify_positions = torch.arange(
            position_ids.item() + 1,
            position_ids.item() + 1 + len(draft_token_ids),
            device=device,
        ).unsqueeze(0)  # (1, K)

        verify_logits = self.verify(verify_input, verify_positions)  # (1, K, V)
        verify_logits = verify_logits.squeeze(0)  # (K, V)

        # verify_logits[i] predicts what should come AFTER draft_tokens[i]
        # So verify_logits[i-1] should predict draft_tokens[i]
        # And verify_logits_0 should predict first_verify_token (which we already used)

        # Build the verify prediction for each draft token position:
        # - Position 0: verify_logits_0 predicts first_verify_token (already matched)
        # - Position i (for i in 1..K-1): verify_logits[i-1] predicts draft_tokens[i]
        verify_predicted = verify_logits.argmax(dim=-1)  # (K,)

        # Compare: draft_tokens[i] should match what verify predicted for position i
        # verify_logits_0 predicted first_verify_token which we used as draft_tokens[0]'s source
        # verify_logits[0] predicts what comes after draft_tokens[0], should match draft_tokens[1]
        # ...
        # verify_logits[K-2] predicts what comes after draft_tokens[K-2], should match draft_tokens[K-1]
        # verify_logits[K-1] predicts what comes after draft_tokens[K-1] = bonus token

        # Check matches: verify_logits[i] predicts token after draft_tokens[i]
        # So for draft_tokens[i+1] to be accepted, verify_logits[i].argmax() must == draft_tokens[i+1]
        accepted = [first_verify_token, draft_token_ids[0]]
        for i in range(len(draft_token_ids) - 1):
            if verify_predicted[i].item() == draft_token_ids[i + 1]:
                accepted.append(draft_token_ids[i + 1])
            else:
                # Mismatch: take verify's prediction as correction
                accepted.append(verify_predicted[i].item())
                break
        else:
            # All draft tokens matched! Add bonus token
            bonus = verify_predicted[-1].item()
            accepted.append(bonus)

        # Rollback verify cache to only include accepted tokens
        # After verify forward: cache has 1 (input) + K (draft) entries
        # We want: initial_pos + n_accepted entries
        # accepted includes: first_verify_token + some draft tokens + correction/bonus
        n_accepted = len(accepted)
        initial_seq_len = position_ids.item()  # seq_len before this step
        self.verify.cache.seq_len = initial_seq_len + n_accepted

        # Clean up draft cache
        self.draft.cache.reset()

        result = torch.tensor(accepted, device=device)
        return result, n_accepted

    @staticmethod
    def accept_tokens(
        draft_tokens: torch.Tensor,
        verify_tokens: torch.Tensor,
        bonus_token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Accept draft tokens up to first mismatch with verify.

        Args:
            draft_tokens: Draft model's predicted tokens (K,).
            verify_tokens: Verify model's predicted tokens (K,).
                verify_tokens[i] is what verify predicts at position i
                (should match draft_tokens[i]).
            bonus_token: Extra token from verify when all K match.

        Returns:
            Tuple of (accepted_tokens, n_accepted).
            On mismatch at position i: accepts draft[0:i] + verify[i].
            On full match: accepts all draft + bonus (if provided).
        """
        k = len(draft_tokens)
        matches = draft_tokens == verify_tokens

        # Find first mismatch
        for i in range(k):
            if not matches[i]:
                # Accept matching prefix + verify's correction
                accepted = list(draft_tokens[:i].tolist()) + [verify_tokens[i].item()]
                return torch.tensor(accepted, device=draft_tokens.device), len(accepted)

        # All matched
        accepted = draft_tokens.tolist()
        if bonus_token is not None:
            accepted.append(bonus_token.item())
        return torch.tensor(accepted, device=draft_tokens.device), len(accepted)


def _make_shared_fp8_step(
    model: torch.nn.Module,
    cache: StaticKVCache,
    max_seq_len: int,
    skip_layers: list[int],
    shared_fp8_weights: dict,
    use_scaled_mm: bool,
    x_scale: torch.Tensor,
) -> FP8GraphableDecodeStep:
    """Create an FP8GraphableDecodeStep that shares FP8 weights from another instance.

    Bypasses _precompute_fp8_weights (weights are already freed from the model).
    Instead, directly assigns the shared fp8_weights dict.
    """
    # Use __new__ to create instance without calling __init__
    step = FP8GraphableDecodeStep.__new__(FP8GraphableDecodeStep)

    # Initialize nn.Module base
    torch.nn.Module.__init__(step)

    # Set up GraphableDecodeStep attributes
    step.model = model
    step.cache = cache
    step.max_seq_len = max_seq_len
    step.skip_layers = set(skip_layers or [])
    step.skip_attention_layers = set()
    step.bridge_layers = {}
    step.expert = None

    config = model.config
    step.num_heads = config.num_attention_heads
    step.num_kv_heads = config.num_key_value_heads
    step.head_dim = config.head_dim
    step.num_kv_groups = step.num_heads // step.num_kv_heads

    # FP8-specific: share weights, don't recompute
    step.fp8_weights = shared_fp8_weights
    step._use_scaled_mm = use_scaled_mm
    step._x_scale = x_scale

    # Pre-compute layer tables (string keys + norm weight references)
    step._precompute_layer_tables()

    # Pre-compute fused QK norm weights (same logic as GraphableDecodeStep.__init__)
    step._fused_qk_norm_weights = []
    step._fused_qk_norm_eps = 1e-6
    for layer in model.model.layers:
        attn = layer.self_attn
        if hasattr(attn, "q_norm") and hasattr(attn, "k_norm"):
            step._fused_qk_norm_eps = attn.q_norm.eps if hasattr(attn.q_norm, "eps") else 1e-6
            q_w = attn.q_norm.weight.data
            k_w = attn.k_norm.weight.data
            fused_w = torch.cat([
                q_w.unsqueeze(0).expand(step.num_heads, -1),
                k_w.unsqueeze(0).expand(step.num_kv_heads, -1),
            ], dim=0)
            step._fused_qk_norm_weights.append(fused_w)
        else:
            step._fused_qk_norm_weights.append(None)

    return step
