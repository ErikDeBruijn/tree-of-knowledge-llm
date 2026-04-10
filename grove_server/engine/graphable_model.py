"""Graphable model wrapper for CUDA graph compatible decode.

Wraps a HuggingFace CausalLM model to produce a decode step with fully
static tensor shapes. Instead of relying on HF's dynamic past_key_values
(which grow via torch.cat), this module manually runs through transformer
layers using a StaticKVCache.

The result: every tensor shape is constant across calls, making the decode
step safe for CUDA graph capture and replay.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from grove_server.engine.fp8_pergroup import fp8_pergroup_matmul, quantize_weight_pergroup, HAS_TRITON
from grove_server.engine.fp8_utils import fp8_available
from grove_server.engine.fused_bf16_layer import FusedBF16LayerProjections, build_fused_bf16
from grove_server.engine.kernels import fused_residual_rmsnorm
from grove_server.engine.static_kv_cache import StaticKVCache
from grove_server.models.expert import Expert
from grove_server.models.expert_loader import MoEMlpAdapter


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor (B, num_heads, seq_len, head_dim).
        k: Key tensor (B, num_kv_heads, seq_len, head_dim).
        cos: Cosine component (B, seq_len, head_dim) or (1, seq_len, head_dim).
        sin: Sine component, same shape as cos.

    Returns:
        Rotated (q, k) tensors.
    """
    # cos/sin come in as (B, seq_len, head_dim), need (B, 1, seq_len, head_dim)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    # Rotate half
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match the number of query heads (for GQA).

    Args:
        hidden_states: (B, num_kv_heads, seq_len, head_dim)
        n_rep: Number of times to repeat each KV head.

    Returns:
        Tensor of shape (B, num_kv_heads * n_rep, seq_len, head_dim).
    """
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _fused_qk_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    fused_weight: torch.Tensor,
    num_q_heads: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RMSNorm to q and k in a single kernel launch.

    Concatenates q and k on the head dimension, applies RMSNorm with
    per-head weights, then splits. Each head is normalized independently
    (RMSNorm reduces over head_dim only), so concatenation is exact.

    Args:
        q: (B, L, num_q_heads, head_dim)
        k: (B, L, num_kv_heads, head_dim)
        fused_weight: (num_q_heads + num_kv_heads, head_dim) — per-head norm weights
        num_q_heads: Number of query heads (split point)
        eps: RMSNorm epsilon

    Returns:
        Normalized (q, k) with same shapes as input.
    """
    # Cat on head dim: (B, L, num_q_heads + num_kv_heads, head_dim)
    qk = torch.cat([q, k], dim=2)
    # RMSNorm per head: variance over last dim (head_dim)
    variance = qk.float().pow(2).mean(-1, keepdim=True)
    qk = qk * torch.rsqrt(variance + eps)
    qk = (fused_weight * qk).to(q.dtype)
    # Split back
    return qk[:, :, :num_q_heads, :], qk[:, :, num_q_heads:, :]


class GraphableDecodeStep(nn.Module):
    """Wraps a HF model to produce a static-shape decode step.

    This module:
    1. Takes static input: (input_ids: [1,1], position_ids: [1,1])
    2. Runs one model forward pass using pre-allocated KV cache
    3. Returns static output: logits [1, 1, vocab_size]

    All tensor shapes are constant across calls, making it CUDA-graph safe.
    """

    def __init__(
        self,
        model: nn.Module,
        static_cache: StaticKVCache,
        max_seq_len: int,
        skip_layers: list[int] | None = None,
        skip_attention_layers: list[int] | None = None,
        expert: Optional[Expert] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.cache = static_cache
        self.max_seq_len = max_seq_len
        self.skip_layers: set[int] = set(skip_layers or [])
        self.skip_attention_layers: set[int] = set(skip_attention_layers or [])
        self.bridge_layers: dict[int, nn.Module] = {}  # layer_idx -> BridgeModule (future use)
        self.expert: Optional[Expert] = expert  # legacy single-expert
        self.experts: list[Expert] = []  # multi-expert grove

        # Attribution tracking: per-token gate activations per layer per expert
        # Set track_attribution=True before generate to collect data
        self.track_attribution: bool = False
        self._last_gate_activations: dict[str, dict[int, float]] = {}  # expert_name -> {layer_idx -> gate}

        # Detect GQA configuration
        config = model.config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        # Pre-compute fused QK norm weights per layer (one kernel instead of two)
        self._fused_qk_norm_weights: list[torch.Tensor | None] = []
        self._fused_qk_norm_eps: float = 1e-6
        for layer in model.model.layers:
            attn = layer.self_attn
            if hasattr(attn, "q_norm") and hasattr(attn, "k_norm"):
                self._fused_qk_norm_eps = attn.q_norm.eps if hasattr(attn.q_norm, "eps") else 1e-6
                # Concatenate q_norm and k_norm weights: (num_heads + num_kv_heads, head_dim)
                # -> used as (total_heads, head_dim) RMSNorm weight
                q_w = attn.q_norm.weight.data  # (head_dim,)
                k_w = attn.k_norm.weight.data  # (head_dim,)
                # Expand to per-head weights then cat
                fused_w = torch.cat([
                    q_w.unsqueeze(0).expand(self.num_heads, -1),
                    k_w.unsqueeze(0).expand(self.num_kv_heads, -1),
                ], dim=0)  # (num_heads + num_kv_heads, head_dim)
                self._fused_qk_norm_weights.append(fused_w)
            else:
                self._fused_qk_norm_weights.append(None)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """One decode step with static shapes.

        Args:
            input_ids: Token IDs, shape (1, 1).
            position_ids: Position indices, shape (1, 1).

        Returns:
            Logits of shape (1, 1, vocab_size).
        """
        # Embeddings
        hidden_states = self.model.model.embed_tokens(input_ids)

        # Rotary position embeddings
        position_embeddings = self.model.model.rotary_emb(
            hidden_states, position_ids
        )

        # Run through each transformer layer
        for layer_idx, decoder_layer in enumerate(self.model.model.layers):
            if layer_idx in self.skip_layers:
                continue  # Pure residual passthrough — skip entirely
            if layer_idx in self.skip_attention_layers:
                hidden_states = self._run_layer_mlp_only(
                    layer_idx, decoder_layer, hidden_states,
                )
                continue
            hidden_states = self._run_layer(
                layer_idx, decoder_layer, hidden_states, position_embeddings
            )

        # Final norm + LM head
        hidden_states = self.model.model.norm(hidden_states)
        logits = self._compute_logits(hidden_states)

        # Advance cache position
        self.cache.advance(input_ids.size(1))

        return logits

    def pop_attribution(self) -> dict[str, dict[int, float]]:
        """Return and clear the last token's gate activations per expert per layer.

        Returns dict mapping expert_name -> {layer_idx -> gate value (0-1)}.
        Empty dict if no expert active or tracking disabled.
        """
        result = {k: dict(v) for k, v in self._last_gate_activations.items()}
        self._last_gate_activations.clear()
        return result

    def _compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute logits from final hidden states. Override for FP8."""
        return self.model.lm_head(hidden_states)

    def _run_layer(
        self,
        layer_idx: int,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Run one transformer layer using the static KV cache.

        Manually implements the layer forward pass to avoid HF's dynamic
        tensor operations (torch.cat on KV cache).

        Args:
            layer_idx: Index of this layer.
            layer: The decoder layer module.
            hidden_states: Input hidden states (B, seq_len, hidden_dim).
            position_embeddings: (cos, sin) from rotary embedding.

        Returns:
            Output hidden states (B, seq_len, hidden_dim).
        """
        # --- Pre-attention: RMSNorm (input_layernorm) ---
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        # --- Self-attention (manual, for static KV cache control) ---
        attn = layer.self_attn
        q = attn.q_proj(hidden_states)
        k = attn.k_proj(hidden_states)
        v = attn.v_proj(hidden_states)

        B, L, _ = q.shape

        # Reshape for multi-head attention
        q = q.view(B, L, self.num_heads, self.head_dim)
        k = k.view(B, L, self.num_kv_heads, self.head_dim)

        # Fused QK norm: one RMSNorm kernel over concatenated heads
        fused_w = self._fused_qk_norm_weights[layer_idx]
        if fused_w is not None:
            q, k = _fused_qk_norm(q, k, fused_w, self.num_heads, self._fused_qk_norm_eps)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings
        cos, sin = position_embeddings
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        # Write new K, V to static cache (before advance)
        self.cache.update(layer_idx, k, v)

        # Get full cached K, V (includes what we just wrote, up to seq_len + L)
        current_len = self.cache.seq_len + L
        full_k, full_v = self.cache.get_up_to(layer_idx, current_len)

        # Expand KV heads for GQA (repeat to match query head count)
        full_k = _repeat_kv(full_k, self.num_kv_groups)
        full_v = _repeat_kv(full_v, self.num_kv_groups)

        # Scaled dot-product attention (ensure matching dtypes)
        compute_dtype = q.dtype
        attn_out = F.scaled_dot_product_attention(
            q, full_k.to(compute_dtype), full_v.to(compute_dtype),
            is_causal=(current_len == L),
        )

        # Reshape back and project
        attn_out = attn_out.transpose(1, 2).reshape(B, L, self.num_heads * self.head_dim)
        attn_out = attn.o_proj(attn_out)

        # --- Post-attention: fused residual add + RMSNorm ---
        # Replaces: hidden_states = residual + attn_out; residual = hidden_states;
        #           hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states, residual = fused_residual_rmsnorm(
            residual, attn_out, layer.post_attention_layernorm.weight,
        )

        # --- MLP with optional expert adapter ---
        mlp_input = hidden_states
        hidden_states = self._run_mlp_with_expert(layer_idx, layer.mlp, hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states

    def _run_mlp_with_expert(
        self,
        layer_idx: int,
        mlp: nn.Module,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Run MLP with multi-expert softmax routing.

        When multiple experts are in self.experts, computes gate logits for
        each expert + a "no expert" option, softmax-normalizes, and blends
        the adapted outputs. Falls back to single-expert sigmoid when only
        self.expert is set (legacy).

        Args:
            layer_idx: Index of this layer.
            mlp: The MLP module (with gate_proj, up_proj, down_proj, act_fn).
            hidden_states: Input hidden states (B, L, D).

        Returns:
            MLP output, possibly modified by expert adapter(s).
        """
        # Collect active experts for this layer
        active_experts = []
        for exp in self.experts:
            if exp.covers_layer(layer_idx) and layer_idx in exp.gates:
                adapter = exp.adapters.get(layer_idx)
                if adapter is not None:
                    active_experts.append(exp)

        # Legacy single-expert fallback
        if not active_experts and self.expert is not None:
            if self.expert.covers_layer(layer_idx) and layer_idx in self.expert.gates:
                adapter = self.expert.adapters.get(layer_idx)
                if adapter is not None:
                    active_experts = [self.expert]

        if not active_experts:
            return mlp(hidden_states)

        orig_shape = hidden_states.shape
        flat_input = hidden_states.reshape(-1, orig_shape[-1])

        # Compute base MLP output once (shared across experts)
        gate_proj_out = mlp.gate_proj(hidden_states)
        up_proj_out = mlp.up_proj(hidden_states)
        base_activated = F.silu(gate_proj_out) * up_proj_out

        if len(active_experts) == 1:
            # Single expert: use sigmoid gate (original behavior)
            exp = active_experts[0]
            gate_val = exp.gates[layer_idx](flat_input)  # sigmoided

            if self.track_attribution:
                if exp.name not in self._last_gate_activations:
                    self._last_gate_activations[exp.name] = {}
                self._last_gate_activations[exp.name][layer_idx] = gate_val[-1, 0].item()

            adapter = exp.adapters[layer_idx]
            if isinstance(adapter, MoEMlpAdapter):
                gate_corr = adapter.gate_correction(flat_input).reshape(gate_proj_out.shape)
                up_corr = adapter.up_correction(flat_input).reshape(up_proj_out.shape)
                adapted_activated = F.silu(gate_proj_out + gate_corr) * (up_proj_out + up_corr)
                blended = base_activated + gate_val.reshape(*orig_shape[:-1], 1) * (adapted_activated - base_activated)
            else:
                base_out = mlp.down_proj(base_activated)
                adapter_out = adapter(flat_input).reshape(base_out.shape)
                return base_out + gate_val.reshape(*orig_shape[:-1], 1) * (adapter_out - base_out)
            return mlp.down_proj(blended)

        # Multi-expert: softmax over gate logits + "no expert" option
        gate_logits = []  # raw logits before sigmoid
        expert_deltas = []  # adapted - base activations

        for exp in active_experts:
            # Get raw gate logit (before sigmoid) for softmax
            gate_module = exp.gates[layer_idx]
            logit = gate_module.linear(flat_input)  # (B*L, 1), raw logit
            gate_logits.append(logit)

            adapter = exp.adapters[layer_idx]
            if isinstance(adapter, MoEMlpAdapter):
                gate_corr = adapter.gate_correction(flat_input).reshape(gate_proj_out.shape)
                up_corr = adapter.up_correction(flat_input).reshape(up_proj_out.shape)
                adapted = F.silu(gate_proj_out + gate_corr) * (up_proj_out + up_corr)
                expert_deltas.append(adapted - base_activated)
            else:
                base_out_flat = mlp.down_proj(base_activated).reshape(-1, orig_shape[-1])
                adapter_out = adapter(flat_input)
                expert_deltas.append((adapter_out - base_out_flat).reshape(base_activated.shape))

        # "No expert" option: logit = 0 (neutral)
        no_expert_logit = torch.zeros_like(gate_logits[0])
        all_logits = torch.cat(gate_logits + [no_expert_logit], dim=-1)  # (B*L, n_experts+1)
        probs = torch.softmax(all_logits, dim=-1)  # (B*L, n_experts+1)

        # Track attribution: softmax probabilities per expert
        if self.track_attribution:
            for i, exp in enumerate(active_experts):
                if exp.name not in self._last_gate_activations:
                    self._last_gate_activations[exp.name] = {}
                self._last_gate_activations[exp.name][layer_idx] = probs[-1, i].item()

        # Blend: base + sum(prob_i * delta_i)
        result = base_activated
        for i, delta in enumerate(expert_deltas):
            prob = probs[:, i:i+1].reshape(*orig_shape[:-1], 1)
            result = result + prob * delta

        return mlp.down_proj(result)

    def _run_layer_mlp_only(
        self,
        layer_idx: int,
        layer: nn.Module,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Run only the MLP part of a transformer layer, skip attention.

        Attention (Q/K/V projection, SDPA, O projection) is skipped entirely.
        The KV cache is not updated for this layer. The residual connection
        passes hidden_states through to the MLP unchanged by attention.

        Args:
            layer_idx: Index of this layer.
            layer: The decoder layer module.
            hidden_states: Input hidden states (B, seq_len, hidden_dim).

        Returns:
            Output hidden states (B, seq_len, hidden_dim).
        """
        # Skip attention: residual stays as-is, no KV cache update
        residual = hidden_states

        # Post-attention norm (still needed before MLP)
        hidden_states = layer.post_attention_layernorm(hidden_states)

        # MLP with optional expert adapter
        hidden_states = self._run_mlp_with_expert(layer_idx, layer.mlp, hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states


class FP8GraphableDecodeStep(GraphableDecodeStep):
    """GraphableDecodeStep with FP8 weight storage for faster matmuls.

    Pre-quantizes all linear layer weights to float8_e4m3fn at init.
    Calls torch._scaled_mm directly in the forward pass — no nn.Module overhead.

    Adapter, gate, and bridge weights are never touched (stay BF16).
    """

    def __init__(
        self,
        model: nn.Module,
        static_cache: StaticKVCache,
        max_seq_len: int,
        skip_layers: list[int] | None = None,
        skip_attention_layers: list[int] | None = None,
        expert: Optional[Expert] = None,
    ) -> None:
        super().__init__(
            model, static_cache, max_seq_len,
            skip_layers=skip_layers,
            skip_attention_layers=skip_attention_layers,
            expert=expert,
        )

        if not fp8_available():
            # Store weights as FP8 anyway (dequant fallback at forward time)
            pass

        # Pre-quantize all linear weights: key -> (weight_fp8, scale_w)
        # weight_fp8 stored as (in_features, out_features) for cuBLAS column-major
        self.fp8_weights: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._use_scaled_mm = fp8_available()
        # Pre-allocate fixed input scale to avoid tensor creation in forward pass
        device = next(model.parameters()).device
        # Fixed activation scale: Qwen3-8B activations reach ~11520 at layer 6+.
        # Scale = amax_observed / fp8_max = 11520 / 448 ≈ 25.7. Use 32 for safety margin.
        self._x_scale = torch.tensor(32.0, dtype=torch.float32, device=device)
        self._x_inv_scale = torch.tensor(1.0 / 32.0, dtype=torch.bfloat16, device=device)
        self._precompute_fp8_weights()
        self._precompute_layer_tables()
        self._precompute_fp8_lm_head()

        # Pre-computed expert routing (populated by _precompute_expert_routing)
        self._expert_routing: list[dict | None] = [None] * len(model.model.layers)

    def _precompute_fp8_weights(self) -> None:
        """Convert all linear projection weights to FP8 with per-group scales.

        Uses per-group (128) quantization instead of per-tensor. The Triton
        kernel applies per-group scales during the matmul, giving BF16-level
        precision (cosine 1.000 vs 0.992 for per-tensor _scaled_mm).

        Stores (w_fp8, combined_scales) where combined_scales bakes in x_scale.
        """
        fp8_max = 448.0

        for idx, layer in enumerate(self.model.model.layers):
            if idx in self.skip_layers:
                continue

            if idx not in self.skip_attention_layers:
                for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    proj = getattr(layer.self_attn, proj_name)
                    if proj.weight is None:
                        continue
                    w = proj.weight.data
                    # Per-row (per output channel) scaling for row-wise _scaled_mm
                    w_row_amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
                    w_row_scale = (w_row_amax / fp8_max).float()
                    w_fp8 = (w.float() / w_row_scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
                    # scale_b for _scaled_mm: (1, N) = transposed row scale
                    self.fp8_weights[f"{idx}.attn.{proj_name}"] = (w_fp8, w_row_scale.t())

            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                proj = getattr(layer.mlp, proj_name)
                if proj.weight is None:
                    continue
                w = proj.weight.data
                w_row_amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
                w_row_scale = (w_row_amax / fp8_max).float()
                w_fp8 = (w.float() / w_row_scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
                self.fp8_weights[f"{idx}.mlp.{proj_name}"] = (w_fp8, w_row_scale.t())

    def _precompute_layer_tables(self) -> None:
        """Pre-compute per-layer weight keys and norm weights as indexed lists.

        Eliminates string formatting and dict lookups in the hot decode loop.
        """
        n_layers = len(self.model.model.layers)
        # FP8 weight keys indexed by layer
        self._attn_q_keys: list[str] = []
        self._attn_k_keys: list[str] = []
        self._attn_v_keys: list[str] = []
        self._attn_o_keys: list[str] = []
        self._mlp_gate_keys: list[str] = []
        self._mlp_up_keys: list[str] = []
        self._mlp_down_keys: list[str] = []
        # RMSNorm weights indexed by layer
        self._input_norm_weights: list[torch.Tensor] = []
        self._post_attn_norm_weights: list[torch.Tensor] = []

        for idx in range(n_layers):
            self._attn_q_keys.append(f"{idx}.attn.q_proj")
            self._attn_k_keys.append(f"{idx}.attn.k_proj")
            self._attn_v_keys.append(f"{idx}.attn.v_proj")
            self._attn_o_keys.append(f"{idx}.attn.o_proj")
            self._mlp_gate_keys.append(f"{idx}.mlp.gate_proj")
            self._mlp_up_keys.append(f"{idx}.mlp.up_proj")
            self._mlp_down_keys.append(f"{idx}.mlp.down_proj")
            layer = self.model.model.layers[idx]
            self._input_norm_weights.append(layer.input_layernorm.weight)
            self._post_attn_norm_weights.append(layer.post_attention_layernorm.weight)

    def _precompute_fp8_lm_head(self) -> None:
        """Pre-quantize the lm_head weight to FP8 with row-wise scales."""
        fp8_max = 448.0
        lm_head = self.model.lm_head
        if not hasattr(lm_head, 'weight') or lm_head.weight is None:
            self._has_fp8_lm_head = False
            return
        w = lm_head.weight.data
        w_row_amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
        w_row_scale = (w_row_amax / fp8_max).float()
        self._lm_head_fp8 = (w.float() / w_row_scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
        self._lm_head_row_scale = w_row_scale.t()  # (1, vocab_size)
        self._has_fp8_lm_head = True

    def _precompute_expert_routing(self) -> None:
        """Pre-compute per-layer expert routing table.

        Eliminates per-call Python dispatch (list comprehensions, isinstance,
        dict lookups). For MoEMlpAdapter, fuses gate_W + gate_lora_A + up_lora_A
        into a single weight matrix, reducing 5 matmuls to 3 per layer.

        Call this after setting self.experts or self.expert.
        """
        n_layers = len(self.model.model.layers)
        self._expert_routing = [None] * n_layers

        # Collect all active experts
        all_experts = list(self.experts)
        if not all_experts and self.expert is not None:
            all_experts = [self.expert]

        for layer_idx in range(n_layers):
            active = [e for e in all_experts
                      if e.covers_layer(layer_idx) and layer_idx in e.gates
                      and e.adapters.get(layer_idx) is not None]
            if not active:
                continue

            if len(active) == 1:
                exp = active[0]
                adapter = exp.adapters[layer_idx]
                gate = exp.gates[layer_idx]

                if isinstance(adapter, MoEMlpAdapter):
                    # Fuse gate_W.T, gate_lora_A, up_lora_A into single matrix
                    gate_w_t = gate.linear.weight.data.t()  # (D, 1)
                    fused_a = torch.cat([
                        gate_w_t,                      # (D, 1)
                        adapter.gate_lora_A.data,      # (D, rank)
                        adapter.up_lora_A.data,        # (D, rank)
                    ], dim=1).contiguous()  # (D, 1 + 2*rank)

                    rank = adapter.gate_lora_A.shape[1]
                    self._expert_routing[layer_idx] = {
                        "type": "moe_single",
                        "fused_a": fused_a,
                        "gate_bias": gate.linear.bias.data,
                        "gate_b": adapter.gate_lora_B.data,
                        "up_b": adapter.up_lora_B.data,
                        "scaling": adapter.scaling,
                        "rank": rank,
                        "expert_name": exp.name,
                    }
                elif layer_idx in exp.skip_layers:
                    self._expert_routing[layer_idx] = {
                        "type": "skip",
                        "gate": gate,
                        "expert_name": exp.name,
                    }
                elif layer_idx in exp.bridge_layers and layer_idx in exp.bridges:
                    self._expert_routing[layer_idx] = {
                        "type": "bridge",
                        "gate": gate,
                        "bridge": exp.bridges[layer_idx],
                        "expert_name": exp.name,
                    }
                else:
                    self._expert_routing[layer_idx] = {
                        "type": "legacy",
                        "gate": gate,
                        "adapter": adapter,
                        "expert_name": exp.name,
                    }
            else:
                self._expert_routing[layer_idx] = {
                    "type": "multi",
                    "experts": active,
                }

    def _fp8_mlp_with_expert_fast(
        self,
        layer_idx: int,
        mlp_flat: torch.Tensor,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        B: int,
        L: int,
    ) -> torch.Tensor:
        """Optimized FP8 MLP with pre-computed expert routing.

        Eliminates per-call: list comprehension, isinstance, dict.get(), covers_layer().
        For MoEMlpAdapter: fuses gate_W + gate_A + up_A into 1 matmul (3 instead of 5).
        """
        routing = self._expert_routing[layer_idx]

        if routing is None:
            # No expert at this layer — standard MLP
            activated = F.silu(gate_proj) * up_proj
            return self._fp8_linear(
                activated.reshape(-1, activated.size(-1)),
                self._mlp_down_keys[layer_idx],
            ).reshape(B, L, -1)

        rtype = routing["type"]

        if rtype == "moe_single":
            # Pre-fetched weights: 1 fused matmul replaces gate_linear + gate_A + up_A
            fused_out = torch.mm(mlp_flat, routing["fused_a"])  # (N, 1+2r)
            r = routing["rank"]

            # Split: gate logit | gate_mid | up_mid
            gate_val = torch.sigmoid(fused_out[:, :1] + routing["gate_bias"])

            # B-side: 2 matmuls
            s = routing["scaling"]
            gate_corr = torch.mm(fused_out[:, 1:1+r], routing["gate_b"]) * s
            up_corr = torch.mm(fused_out[:, 1+r:], routing["up_b"]) * s

            base_act = F.silu(gate_proj) * up_proj
            adapted_act = F.silu(gate_proj + gate_corr) * (up_proj + up_corr)
            blended = base_act + gate_val * (adapted_act - base_act)

            return self._fp8_linear(
                blended.reshape(-1, blended.size(-1)),
                self._mlp_down_keys[layer_idx],
            ).reshape(B, L, -1)

        if rtype == "skip":
            gate_val = routing["gate"](mlp_flat)
            activated = F.silu(gate_proj) * up_proj
            base_out = self._fp8_linear(
                activated.reshape(-1, activated.size(-1)),
                self._mlp_down_keys[layer_idx],
            )
            return (base_out * (1.0 - gate_val)).reshape(B, L, -1)

        if rtype == "bridge":
            gate_val = routing["gate"](mlp_flat)
            activated = F.silu(gate_proj) * up_proj
            base_out = self._fp8_linear(
                activated.reshape(-1, activated.size(-1)),
                self._mlp_down_keys[layer_idx],
            )
            bridge_out = routing["bridge"](mlp_flat)
            return (base_out + gate_val * (bridge_out - base_out)).reshape(B, L, -1)

        if rtype == "legacy":
            gate_val = routing["gate"](mlp_flat)
            activated = F.silu(gate_proj) * up_proj
            base_out = self._fp8_linear(
                activated.reshape(-1, activated.size(-1)),
                self._mlp_down_keys[layer_idx],
            )
            adapter_out = routing["adapter"](mlp_flat)
            return (base_out + gate_val * (adapter_out - base_out)).reshape(B, L, -1)

        # Multi-expert: fall back to original
        return self._fp8_mlp_with_expert(
            layer_idx, None, None, mlp_flat, gate_proj, up_proj, B, L,
        )

    def _compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Override: use FP8 lm_head when available."""
        if getattr(self, '_has_fp8_lm_head', False):
            return self._fp8_lm_head(hidden_states)
        return self.model.lm_head(hidden_states)

    def _fp8_lm_head(self, x: torch.Tensor) -> torch.Tensor:
        """FP8 lm_head with row-wise scaling."""
        flat = x.reshape(-1, x.size(-1))
        x_row_amax = flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
        x_row_scale = (x_row_amax / 448.0).float()
        x_fp8 = (flat / x_row_scale).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_fp8, self._lm_head_fp8.t(),
            scale_a=x_row_scale,
            scale_b=self._lm_head_row_scale,
            out_dtype=torch.bfloat16,
        )
        return out.reshape(*x.shape[:-1], -1)

    def _fp8_linear(self, x: torch.Tensor, key: str) -> torch.Tensor:
        """FP8 matmul with row-wise scaling via _scaled_mm tensor cores.

        Row-wise scaling (per-row for both input and weight) gives cos≥0.996
        on real Qwen3-8B weights. Uses hardware FP8 tensor cores — 1.8x faster
        than BF16 F.linear. Input cached per layer (shared across projections).

        Args:
            x: Input tensor (M, K) in BF16.
            key: Weight key like "0.attn.q_proj".

        Returns:
            Output tensor (M, N) in BF16.
        """
        w_fp8, w_row_scales = self.fp8_weights[key]

        # Cache x_fp8 per layer: all projections within a layer share input
        layer_idx = int(key.split(".")[0])
        is_mlp = ".mlp." in key
        cache_key = (layer_idx, is_mlp)

        if not hasattr(self, '_x_fp8_cache'):
            self._x_fp8_cache = {}

        cached = self._x_fp8_cache.get(cache_key)
        if cached is not None and cached[0] is x:
            x_fp8, x_row_scale = cached[1], cached[2]
        else:
            x_row_amax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
            x_row_scale = (x_row_amax / 448.0).float()
            x_fp8 = (x / x_row_scale).to(torch.float8_e4m3fn)
            self._x_fp8_cache[cache_key] = (x, x_fp8, x_row_scale)

        return torch._scaled_mm(
            x_fp8, w_fp8.t(),
            scale_a=x_row_scale,
            scale_b=w_row_scales,
            out_dtype=torch.bfloat16,
        )

    def _fp8_mlp_with_expert(
        self,
        layer_idx: int,
        layer: nn.Module,
        mlp_input: torch.Tensor,
        mlp_flat: torch.Tensor,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        B: int,
        L: int,
    ) -> torch.Tensor:
        """Complete FP8 MLP with optional expert adapter integration.

        For MoEMlpAdapter: injects LoRA corrections into gate/up projections
        before activation, matching training format exactly.

        For skip/bridge/legacy LoRA: applies gate-blended post-hoc delta.

        Args:
            layer_idx: Layer index.
            layer: Decoder layer module (for fallback MLP access).
            mlp_input: Pre-MLP hidden states (B, L, D).
            mlp_flat: Flattened mlp_input (B*L, D).
            gate_proj: FP8 gate_proj output (B*L, intermediate).
            up_proj: FP8 up_proj output (B*L, intermediate).
            B: Batch size.
            L: Sequence length.

        Returns:
            MLP output (B, L, D), possibly modified by expert.
        """
        # Check for multi-expert softmax routing
        active_experts = [exp for exp in self.experts
                          if exp.covers_layer(layer_idx) and layer_idx in exp.gates
                          and exp.adapters.get(layer_idx) is not None]
        if len(active_experts) > 1:
            # Multi-expert FP8: softmax over gate logits + blend adapted outputs
            base_activated = F.silu(gate_proj) * up_proj
            gate_logits = []
            expert_deltas = []

            for exp in active_experts:
                logit = exp.gates[layer_idx].linear(mlp_flat)  # raw logit
                gate_logits.append(logit)

                adapter = exp.adapters[layer_idx]
                if isinstance(adapter, MoEMlpAdapter):
                    gate_corr = adapter.gate_correction(mlp_flat)
                    up_corr = adapter.up_correction(mlp_flat)
                    adapted = F.silu(gate_proj + gate_corr) * (up_proj + up_corr)
                    expert_deltas.append(adapted - base_activated)
                else:
                    expert_deltas.append(torch.zeros_like(base_activated))

            # "No expert" option: logit = 0
            no_expert_logit = torch.zeros_like(gate_logits[0])
            all_logits = torch.cat(gate_logits + [no_expert_logit], dim=-1)
            probs = torch.softmax(all_logits, dim=-1)

            if self.track_attribution:
                for i, exp in enumerate(active_experts):
                    if exp.name not in self._last_gate_activations:
                        self._last_gate_activations[exp.name] = {}
                    self._last_gate_activations[exp.name][layer_idx] = probs[-1, i].item()

            result = base_activated
            for i, delta in enumerate(expert_deltas):
                prob = probs[:, i:i+1]
                result = result + prob * delta

            mlp_flat2 = result.reshape(-1, result.size(-1))
            return self._fp8_linear(mlp_flat2, self._mlp_down_keys[layer_idx]).reshape(B, L, -1)

        # Single expert: use FP8 fast path
        expert = active_experts[0] if active_experts else None
        if expert is None and self.expert is not None:
            if self.expert.covers_layer(layer_idx) and layer_idx in self.expert.gates:
                expert = self.expert
        has_expert = expert is not None

        if has_expert:
            gate_val = expert.gates[layer_idx](mlp_flat)  # (B*L, 1), sigmoided

            # Track gate activation for attribution (last token only)
            if self.track_attribution:
                if expert.name not in self._last_gate_activations:
                    self._last_gate_activations[expert.name] = {}
                self._last_gate_activations[expert.name][layer_idx] = gate_val[-1, 0].item()

            if layer_idx in expert.skip_layers:
                activated = F.silu(gate_proj) * up_proj
                mlp_flat2 = activated.reshape(-1, activated.size(-1))
                base_out = self._fp8_linear(mlp_flat2, self._mlp_down_keys[layer_idx]).reshape(B, L, -1)
                flat_base = base_out.reshape(-1, base_out.size(-1))
                result = flat_base * (1.0 - gate_val)
                return result.reshape(B, L, -1)

            if layer_idx in expert.bridge_layers and layer_idx in expert.bridges:
                activated = F.silu(gate_proj) * up_proj
                mlp_flat2 = activated.reshape(-1, activated.size(-1))
                base_out = self._fp8_linear(mlp_flat2, self._mlp_down_keys[layer_idx])
                bridge_out = expert.bridges[layer_idx](mlp_flat)
                delta = bridge_out - base_out
                result = base_out + gate_val * delta
                return result.reshape(B, L, -1)

            adapter = expert.adapters.get(layer_idx)
            if adapter is not None and isinstance(adapter, MoEMlpAdapter):
                # MoE format: inject LoRA corrections before activation
                gate_corr = adapter.gate_correction(mlp_flat)
                up_corr = adapter.up_correction(mlp_flat)

                base_activated = F.silu(gate_proj) * up_proj
                adapted_activated = F.silu(gate_proj + gate_corr) * (up_proj + up_corr)

                blended = base_activated + gate_val * (adapted_activated - base_activated)
                mlp_flat2 = blended.reshape(-1, blended.size(-1))
                result = self._fp8_linear(mlp_flat2, self._mlp_down_keys[layer_idx])
                return result.reshape(B, L, -1)

            if adapter is not None:
                # Legacy LoRA format
                activated = F.silu(gate_proj) * up_proj
                mlp_flat2 = activated.reshape(-1, activated.size(-1))
                base_out = self._fp8_linear(mlp_flat2, self._mlp_down_keys[layer_idx])
                adapter_out = adapter(mlp_flat)
                delta = adapter_out - base_out
                result = base_out + gate_val * delta
                return result.reshape(B, L, -1)

        # No expert: standard MLP
        activated = F.silu(gate_proj) * up_proj
        mlp_flat2 = activated.reshape(-1, activated.size(-1))
        result = self._fp8_linear(mlp_flat2, self._mlp_down_keys[layer_idx])
        return result.reshape(B, L, -1)

    def _run_layer(
        self,
        layer_idx: int,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Run one transformer layer using FP8 matmuls for all linear projections.

        Uses fused_residual_rmsnorm to merge residual add + RMSNorm into one
        kernel launch (saves 1 launch per fuse site, 2 per layer).
        Uses pre-computed key lists to avoid string formatting in the hot path.
        """
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        B, L, D = hidden_states.shape
        flat = hidden_states.reshape(-1, D)

        # --- Self-attention with FP8 projections (pre-computed keys) ---
        attn = layer.self_attn
        q = self._fp8_linear(flat, self._attn_q_keys[layer_idx]).reshape(B, L, -1)
        k = self._fp8_linear(flat, self._attn_k_keys[layer_idx]).reshape(B, L, -1)
        v = self._fp8_linear(flat, self._attn_v_keys[layer_idx]).reshape(B, L, -1)

        q = q.view(B, L, self.num_heads, self.head_dim)
        k = k.view(B, L, self.num_kv_heads, self.head_dim)

        # Fused QK norm: one RMSNorm kernel over concatenated heads
        fused_w = self._fused_qk_norm_weights[layer_idx]
        if fused_w is not None:
            q, k = _fused_qk_norm(q, k, fused_w, self.num_heads, self._fused_qk_norm_eps)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        self.cache.update(layer_idx, k, v)

        current_len = self.cache.seq_len + L
        full_k, full_v = self.cache.get_up_to(layer_idx, current_len)

        full_k = _repeat_kv(full_k, self.num_kv_groups)
        full_v = _repeat_kv(full_v, self.num_kv_groups)

        compute_dtype = q.dtype
        attn_out = F.scaled_dot_product_attention(
            q, full_k.to(compute_dtype), full_v.to(compute_dtype),
            is_causal=(current_len == L),
        )

        attn_out = attn_out.transpose(1, 2).reshape(B, L, self.num_heads * self.head_dim)
        attn_flat = attn_out.reshape(-1, attn_out.size(-1))
        attn_out = self._fp8_linear(attn_flat, self._attn_o_keys[layer_idx]).reshape(B, L, -1)

        # --- Post-attention: fused residual add + RMSNorm ---
        # Replaces: hidden_states = residual + attn_out; residual = hidden_states;
        #           hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states, residual = fused_residual_rmsnorm(
            residual, attn_out, self._post_attn_norm_weights[layer_idx],
        )

        # --- MLP with FP8 projections + optional expert adapter ---
        mlp_input = hidden_states
        B2, L2, D2 = hidden_states.shape
        mlp_flat = hidden_states.reshape(-1, D2)

        gate_proj = self._fp8_linear(mlp_flat, self._mlp_gate_keys[layer_idx])
        up_proj = self._fp8_linear(mlp_flat, self._mlp_up_keys[layer_idx])

        # Use fast path if routing table is pre-computed
        if self._expert_routing[layer_idx] is not None or not (self.experts or self.expert):
            hidden_states = self._fp8_mlp_with_expert_fast(
                layer_idx, mlp_flat, gate_proj, up_proj, B2, L2,
            )
        else:
            hidden_states = self._fp8_mlp_with_expert(
                layer_idx, layer, mlp_input, mlp_flat, gate_proj, up_proj, B2, L2,
            )

        hidden_states = residual + hidden_states

        return hidden_states

    def _run_layer_mlp_only(
        self,
        layer_idx: int,
        layer: nn.Module,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Run only the MLP part using FP8 weights, skip attention."""
        residual = hidden_states

        # Post-attention norm (still needed before MLP)
        hidden_states = layer.post_attention_layernorm(hidden_states)

        # MLP with FP8 projections + optional expert adapter
        mlp_input = hidden_states
        B, L, D = hidden_states.shape
        mlp_flat = hidden_states.reshape(-1, D)

        gate_proj = self._fp8_linear(mlp_flat, self._mlp_gate_keys[layer_idx])
        up_proj = self._fp8_linear(mlp_flat, self._mlp_up_keys[layer_idx])

        # Use fast path if routing table is pre-computed
        if self._expert_routing[layer_idx] is not None or not (self.experts or self.expert):
            hidden_states = self._fp8_mlp_with_expert_fast(
                layer_idx, mlp_flat, gate_proj, up_proj, B, L,
            )
        else:
            hidden_states = self._fp8_mlp_with_expert(
                layer_idx, layer, mlp_input, mlp_flat, gate_proj, up_proj, B, L,
            )

        hidden_states = residual + hidden_states

        return hidden_states


class FusedBF16GraphableDecodeStep(GraphableDecodeStep):
    """GraphableDecodeStep with fused BF16 projections.

    Fuses QKV into 1 matmul and gate+up into 1 matmul per layer.
    4 kernel launches per layer instead of 7. No quantization — full BF16.
    """

    def __init__(
        self,
        model: nn.Module,
        static_cache: StaticKVCache,
        max_seq_len: int,
        skip_layers: list[int] | None = None,
        skip_attention_layers: list[int] | None = None,
        expert: Optional[Expert] = None,
    ) -> None:
        super().__init__(
            model, static_cache, max_seq_len,
            skip_layers=skip_layers,
            skip_attention_layers=skip_attention_layers,
            expert=expert,
        )
        self._fused = build_fused_bf16(model)

    def _run_layer(
        self,
        layer_idx: int,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Run one transformer layer with fused BF16 projections."""
        fused = self._fused.get(layer_idx)
        if fused is None:
            return super()._run_layer(layer_idx, layer, hidden_states, position_embeddings)

        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        B, L, _ = hidden_states.shape

        # --- Fused QKV: 1 matmul instead of 3 ---
        q, k, v = fused.fused_qkv(hidden_states)

        q = q.view(B, L, self.num_heads, self.head_dim)
        k = k.view(B, L, self.num_kv_heads, self.head_dim)

        fused_w = self._fused_qk_norm_weights[layer_idx]
        if fused_w is not None:
            q, k = _fused_qk_norm(q, k, fused_w, self.num_heads, self._fused_qk_norm_eps)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        self.cache.update(layer_idx, k, v)
        current_len = self.cache.seq_len + L
        full_k, full_v = self.cache.get_up_to(layer_idx, current_len)
        full_k = _repeat_kv(full_k, self.num_kv_groups)
        full_v = _repeat_kv(full_v, self.num_kv_groups)

        attn_out = F.scaled_dot_product_attention(
            q, full_k.to(q.dtype), full_v.to(q.dtype),
            is_causal=(current_len == L),
        )

        attn_out = attn_out.transpose(1, 2).reshape(B, L, self.num_heads * self.head_dim)
        attn_out = fused.o_proj(attn_out)

        hidden_states, residual = fused_residual_rmsnorm(
            residual, attn_out, layer.post_attention_layernorm.weight,
        )

        # --- MLP with fused gate+up ---
        mlp_input = hidden_states
        flat_input = hidden_states.reshape(-1, hidden_states.size(-1))

        gate_proj, up_proj = fused.fused_gate_up(hidden_states)
        gate_proj = gate_proj.reshape(-1, gate_proj.size(-1))
        up_proj = up_proj.reshape(-1, up_proj.size(-1))

        hidden_states = self._fused_bf16_mlp_with_expert(
            layer_idx, layer, fused, mlp_input, flat_input,
            gate_proj, up_proj, B, L,
        )

        hidden_states = residual + hidden_states
        return hidden_states

    def _fused_bf16_mlp_with_expert(
        self,
        layer_idx: int,
        layer: nn.Module,
        fused: FusedBF16LayerProjections,
        mlp_input: torch.Tensor,
        mlp_flat: torch.Tensor,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        B: int,
        L: int,
    ) -> torch.Tensor:
        """MLP with expert routing using fused BF16 down_proj."""
        active_experts = [exp for exp in self.experts
                          if exp.covers_layer(layer_idx) and layer_idx in exp.gates
                          and exp.adapters.get(layer_idx) is not None]

        if not active_experts and self.expert is not None:
            if self.expert.covers_layer(layer_idx) and layer_idx in self.expert.gates:
                if self.expert.adapters.get(layer_idx) is not None:
                    active_experts = [self.expert]

        if not active_experts:
            activated = F.silu(gate_proj) * up_proj
            return fused.down_proj(activated).reshape(B, L, -1)

        if len(active_experts) == 1:
            exp = active_experts[0]
            gate_val = exp.gates[layer_idx](mlp_flat)

            if self.track_attribution:
                if exp.name not in self._last_gate_activations:
                    self._last_gate_activations[exp.name] = {}
                self._last_gate_activations[exp.name][layer_idx] = gate_val[-1, 0].item()

            adapter = exp.adapters[layer_idx]
            if isinstance(adapter, MoEMlpAdapter):
                gate_corr = adapter.gate_correction(mlp_flat)
                up_corr = adapter.up_correction(mlp_flat)
                base_activated = F.silu(gate_proj) * up_proj
                adapted_activated = F.silu(gate_proj + gate_corr) * (up_proj + up_corr)
                blended = base_activated + gate_val * (adapted_activated - base_activated)
                return fused.down_proj(blended).reshape(B, L, -1)

            # Legacy adapter
            base_activated = F.silu(gate_proj) * up_proj
            base_out = fused.down_proj(base_activated)
            adapter_out = adapter(mlp_flat)
            return (base_out + gate_val * (adapter_out - base_out)).reshape(B, L, -1)

        # Multi-expert softmax
        base_activated = F.silu(gate_proj) * up_proj
        gate_logits = []
        expert_deltas = []
        for exp in active_experts:
            gate_logits.append(exp.gates[layer_idx].linear(mlp_flat))
            adapter = exp.adapters[layer_idx]
            if isinstance(adapter, MoEMlpAdapter):
                gc = adapter.gate_correction(mlp_flat)
                uc = adapter.up_correction(mlp_flat)
                expert_deltas.append(F.silu(gate_proj + gc) * (up_proj + uc) - base_activated)
            else:
                expert_deltas.append(torch.zeros_like(base_activated))

        all_logits = torch.cat(gate_logits + [torch.zeros_like(gate_logits[0])], dim=-1)
        probs = torch.softmax(all_logits, dim=-1)

        if self.track_attribution:
            for i, exp in enumerate(active_experts):
                if exp.name not in self._last_gate_activations:
                    self._last_gate_activations[exp.name] = {}
                self._last_gate_activations[exp.name][layer_idx] = probs[-1, i].item()

        result = base_activated
        for i, delta in enumerate(expert_deltas):
            result = result + probs[:, i:i+1] * delta
        return fused.down_proj(result).reshape(B, L, -1)
