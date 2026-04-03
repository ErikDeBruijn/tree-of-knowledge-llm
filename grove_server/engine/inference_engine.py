"""InferenceEngine: wraps a HuggingFace model with expert injection."""

from __future__ import annotations

import logging
import types
from typing import Iterator, Optional

import torch

logger = logging.getLogger(__name__)
from transformers import AutoModelForCausalLM, AutoTokenizer

from grove_server.engine.cuda_graph import CUDAGraphRunner
from grove_server.engine.fp8_utils import fp8_available
from grove_server.engine.graphable_model import FP8GraphableDecodeStep, GraphableDecodeStep
from grove_server.engine.int4_layer import Int4PackedLinear, quantize_model, set_all_precision
from grove_server.engine.layer_executor import execute_layer
from grove_server.engine.static_kv_cache import StaticKVCache
from grove_server.models.expert import Expert


class InferenceEngine:
    """Load a base model and optionally inject expert behavior into its layers.

    The engine hooks into model.model.layers[l].forward to intercept
    hidden states and route them through the expert's adapters, gates,
    bridges, and skip logic via ``execute_layer``.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: str = "bfloat16",
        skip_layers: list[int] | None = None,
        disable_fast_pipeline: bool = False,
        quantization: str | None = None,
    ) -> None:
        """Load base model and tokenizer.

        Args:
            model_name: HuggingFace model identifier.
            device: Target device ("cpu", "cuda", "auto").
            dtype: Weight dtype ("bfloat16", "float16", "float32").
            skip_layers: Layer indices to skip (residual passthrough).
        """
        self._skip_layers = skip_layers or []
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        # "auto" → single GPU (required for fast pipeline / CUDA graph)
        if device == "auto" and torch.cuda.is_available():
            device_map = {"": 0}
        elif device == "auto":
            device_map = "auto"
        else:
            device_map = None

        load_kwargs = {"device_map": device_map}
        if quantization == "nf4":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
            )
            logger.info("Loading model in NF4 quantization")
        else:
            load_kwargs["torch_dtype"] = torch_dtype

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if device_map is None:
            self.model = self.model.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device if device not in ("auto",) else str(self.model.device)
        self.num_layers: int = self.model.config.num_hidden_layers

        self._active_expert: Optional[Expert] = None
        self._original_forwards: dict[int, types.MethodType] = {}
        self._graph_runner: Optional[CUDAGraphRunner] = None
        self._graphable: Optional[GraphableDecodeStep] = None
        self._static_cache: Optional[StaticKVCache] = None

        # Persistent CUDA graph for decode (captured once, reused across requests)
        self._decode_graph: Optional[torch.cuda.CUDAGraph] = None
        self._decode_static_tok: Optional[torch.Tensor] = None
        self._decode_static_pos: Optional[torch.Tensor] = None
        self._decode_static_logits: Optional[torch.Tensor] = None

        # Auto-build fast pipeline on CUDA when model is on a single device.
        # device_map="auto" can split across GPUs, which GraphableDecodeStep
        # doesn't support (all tensors must be on the same device).
        # INT4 packed quantization: gate-informed variable precision
        self._int4_registry: dict | None = None
        if quantization == "int4_packed":
            self._int4_registry = quantize_model(self.model, start_layer=0)
            logger.info("INT4 packed quantization: %d layers, gate-informed precision",
                        len(self._int4_registry))
        elif quantization:
            logger.info("Using HF generate (quantization=%s, no graphable step)", quantization)

        if not quantization and "cuda" in str(self.device) and self._model_on_single_device():
            try:
                if disable_fast_pipeline:
                    self._build_bf16_pipeline()
                    logger.info("BF16 fast pipeline built (training-compatible)")
                else:
                    self._build_fast_pipeline()
                    logger.info("Fast pipeline built (skip_layers=%s)", self._skip_layers)
            except Exception as e:
                logger.warning("Fast pipeline build failed: %s", e)

    def _model_on_single_device(self) -> bool:
        """Check if all model parameters reside on the same device."""
        devices = {str(p.device) for p in self.model.parameters()}
        return len(devices) == 1

    @property
    def _fast_pipeline_available(self) -> bool:
        """True when graphable decode step and static cache are ready."""
        return self._graphable is not None and self._static_cache is not None

    def _build_graphable_decode(self, max_seq_len: int = 2048) -> None:
        """Build a CUDA-graphable decode step with static KV cache.

        Creates a StaticKVCache and GraphableDecodeStep that can be used
        with CUDAGraphRunner for zero-overhead decode.

        Args:
            max_seq_len: Maximum sequence length to pre-allocate.
        """
        config = self.model.config
        use_fp8 = fp8_available()
        self._static_cache = StaticKVCache(
            num_layers=config.num_hidden_layers,
            num_heads=config.num_key_value_heads,
            head_dim=getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads),
            max_seq_len=max_seq_len,
            batch_size=1,
            dtype=next(self.model.parameters()).dtype,
            device=self.device,
            kv_dtype=torch.float8_e4m3fn if use_fp8 else None,
        )
        if use_fp8:
            self._graphable = FP8GraphableDecodeStep(
                self.model, self._static_cache, max_seq_len=max_seq_len
            )
        else:
            self._graphable = GraphableDecodeStep(
                self.model, self._static_cache, max_seq_len=max_seq_len
            )
        self._graph_runner = CUDAGraphRunner(device=self.device)

    def _build_bf16_pipeline(self, max_seq_len: int = 2048) -> None:
        """Build BF16 fast pipeline that preserves original weights.

        Used when training is active — FP8 sets proj.weight=None to save VRAM
        which breaks training hooks. BF16 graphable step is slower than FP8
        but still faster than naive HF forward, and keeps weights intact.
        """
        config = self.model.config
        self._static_cache = StaticKVCache(
            num_layers=config.num_hidden_layers,
            num_heads=config.num_key_value_heads,
            head_dim=getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads),
            max_seq_len=max_seq_len,
            batch_size=1,
            dtype=next(self.model.parameters()).dtype,
            device=self.device,
        )
        self._graphable = GraphableDecodeStep(
            self.model, self._static_cache, max_seq_len=max_seq_len,
            skip_layers=self._skip_layers,
        )

    def _build_fast_pipeline(self, max_seq_len: int = 2048) -> None:
        """Build fast inference pipeline with GraphableDecodeStep + StaticKVCache.

        Tries FP8 first (faster on Hopper/Blackwell), falls back to BF16.

        Args:
            max_seq_len: Maximum sequence length to pre-allocate.
        """
        config = self.model.config
        self._static_cache = StaticKVCache(
            num_layers=config.num_hidden_layers,
            num_heads=config.num_key_value_heads,
            head_dim=getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads),
            max_seq_len=max_seq_len,
            batch_size=1,
            dtype=next(self.model.parameters()).dtype,
            device=self.device,
        )
        # Try FP8 (with fixed activation scale to avoid NaN on large activations)
        if fp8_available():
            try:
                self._graphable = FP8GraphableDecodeStep(
                    self.model, self._static_cache, max_seq_len=max_seq_len,
                    skip_layers=self._skip_layers,
                )
                # Quick sanity check: run one forward pass
                test_ids = torch.zeros(1, 1, dtype=torch.long, device=self.device)
                pos = torch.zeros(1, 1, dtype=torch.long, device=self.device)
                self._static_cache.reset()
                with torch.no_grad():
                    test_out = self._graphable(test_ids, pos)
                if not test_out.isnan().any():
                    return  # FP8 works!
                # FP8 produced NaN, fall through to BF16
                self._graphable = None
            except Exception:
                self._graphable = None

        # Fallback: BF16
        self._graphable = GraphableDecodeStep(
            self.model, self._static_cache, max_seq_len=max_seq_len,
            skip_layers=self._skip_layers,
        )

    def _ensure_decode_graph(self, sample_token: torch.Tensor) -> None:
        """CUDA graph for decode is disabled — KV cache uses Python-level seq_len
        for indexing, which is not captured by CUDA graphs. This causes all decode
        steps to write to the same cache position, producing repetitive output.

        TODO: To re-enable, make cache position a tensor (part of the graph).
        """
        pass

    def _invalidate_graph(self) -> None:
        """Invalidate any captured CUDA graph (expert config changed)."""
        if self._graph_runner is not None:
            self._graph_runner.invalidate()
            self._graph_runner = None
        # Also invalidate persistent decode graph
        self._decode_graph = None
        self._decode_static_tok = None
        self._decode_static_pos = None
        self._decode_static_logits = None

    def _build_decode_fn(self):
        """Build a callable that runs one decode step through the model.

        Returns a function: hidden_states (1, 1, D) -> logits (1, 1, V)
        that runs the full model forward with KV cache.
        """
        model = self.model

        def decode_fn(input_ids_and_cache):
            # input_ids_and_cache is just input_ids for the current token
            outputs = model(input_ids_and_cache)
            return outputs.logits[:, -1:, :]

        return decode_fn

    def _get_or_capture_graph(self, sample_input: torch.Tensor):
        """Get existing graph runner or capture a new one.

        Returns the CUDAGraphRunner (captured and ready for replay).
        """
        if self._graph_runner is not None and self._graph_runner.is_captured:
            return self._graph_runner

        runner = CUDAGraphRunner(device=self.device)
        decode_fn = self._build_decode_fn()
        runner.capture(decode_fn, sample_input)
        self._graph_runner = runner
        return runner

    def install_expert(self, expert: Expert) -> None:
        """Hook into model layers to inject expert behavior.

        If another expert is already installed, it is uninstalled first.
        For each layer in the expert's range that has a gate, the layer's
        forward method is replaced with a wrapper that calls
        ``execute_layer``.
        """
        if self._active_expert is not None:
            self.uninstall_expert()

        self._invalidate_graph()
        layers = self.model.model.layers

        for layer_idx in range(expert.start_layer, expert.end_layer):
            if layer_idx >= len(layers):
                break
            if layer_idx not in expert.gates:
                continue

            layer = layers[layer_idx]
            self._original_forwards[layer_idx] = layer.forward

            # Build the hooked forward — capture layer_idx, expert, and
            # original forward in the closure.
            original_fwd = layer.forward

            def _make_hook(
                l_idx: int, orig_fwd: types.MethodType, exp: Expert
            ):
                def hooked_forward(hidden_states, **kwargs):
                    result = execute_layer(
                        layer_idx=l_idx,
                        hidden_states=hidden_states,
                        expert=exp,
                        base_layer=lambda x: orig_fwd(x, **kwargs),
                    )
                    return result
                return hooked_forward

            layer.forward = _make_hook(layer_idx, original_fwd, expert)

        self._active_expert = expert

        # Also install into the fast pipeline (GraphableDecodeStep)
        if self._graphable is not None:
            self._graphable.expert = expert

    def uninstall_expert(self) -> None:
        """Restore original model forwards."""
        if self._active_expert is None:
            return

        self._invalidate_graph()
        layers = self.model.model.layers
        for layer_idx, original_fwd in self._original_forwards.items():
            if layer_idx < len(layers):
                layers[layer_idx].forward = original_fwd

        self._original_forwards.clear()
        self._active_expert = None

        # Also clear from fast pipeline
        if self._graphable is not None:
            self._graphable.expert = None
            self._graphable.experts = []

    def install_experts(self, experts: list[Expert]) -> None:
        """Install multiple experts for simultaneous softmax routing.

        Uses the GraphableDecodeStep multi-expert path.
        """
        self.uninstall_expert()
        if self._graphable is not None:
            self._graphable.experts = experts
        logger.info("Installed %d experts: %s", len(experts), [e.name for e in experts])

    def generate_with_attribution(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> list[dict]:
        """Generate with per-token, per-expert attribution.

        Returns list of dicts:
          [{"token": "text", "expert_gates": {"pubmed": {12: 0.8, ...}, "legal": {12: 0.3, ...}}}]

        When only one expert is active, also populates "layer_gates" for backwards compat.
        """
        if not self._fast_pipeline_available:
            text = self._generate_naive(prompt, max_tokens, temperature)
            tokens = self.tokenizer.encode(text)
            return [{"token": self.tokenizer.decode([t]), "layer_gates": {}, "expert_gates": {}} for t in tokens]

        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        self._static_cache.reset()
        self._graphable.track_attribution = True

        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        result: list[dict] = []

        with torch.no_grad():
            # Prefill
            pos = torch.arange(input_ids.size(1), device=self.device).unsqueeze(0)
            logits = self._graphable(input_ids, pos)
            self._graphable.pop_attribution()  # discard prefill attribution

            next_token = self._sample_token(logits[:, -1, :], temperature)

            for _ in range(max_tokens):
                tok_id = next_token.item()
                if eos_id is not None and tok_id == eos_id:
                    break

                pos = torch.tensor([[self._static_cache.seq_len]], device=self.device)
                logits = self._graphable(next_token.unsqueeze(0) if next_token.dim() == 1 else next_token, pos)
                attribution = self._graphable.pop_attribution()

                token_str = self.tokenizer.decode([tok_id])
                # attribution is now {expert_name: {layer: gate, ...}, ...}
                # For backwards compat, flatten first expert into layer_gates
                flat_gates = {}
                if attribution:
                    first_expert = next(iter(attribution.values()), {})
                    flat_gates = first_expert
                result.append({
                    "token": token_str,
                    "layer_gates": flat_gates,
                    "expert_gates": attribution,
                })

                next_token = self._sample_token(logits[:, -1, :], temperature)

        self._graphable.track_attribution = False
        return result

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Generate text completion. Uses fast pipeline on CUDA, naive on CPU.

        Args:
            prompt: Input text.
            max_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text (excluding the prompt).
        """
        if self._fast_pipeline_available:
            return self._generate_fast(prompt, max_tokens, temperature)
        return self._generate_naive(prompt, max_tokens, temperature)

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        use_cuda_graph: bool = False,
    ) -> Iterator[str]:
        """Streaming generation. Uses fast pipeline on CUDA, naive on CPU.

        Args:
            prompt: Input text.
            max_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature.
            use_cuda_graph: Legacy parameter, ignored (CUDA graphs are used
                automatically when the fast pipeline is available).
        """
        if self._fast_pipeline_available:
            yield from self._generate_stream_fast(prompt, max_tokens, temperature)
        else:
            yield from self._generate_stream_naive(prompt, max_tokens, temperature)

    # ------------------------------------------------------------------
    # Naive (O(n^2)) methods — CPU fallback
    # ------------------------------------------------------------------

    def _generate_naive(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Naive generate using HF model.generate (no KV cache optimization)."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        prompt_len = input_ids.shape[1]

        gen_kwargs: dict = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
        }

        with torch.no_grad():
            output_ids = self.model.generate(input_ids, **gen_kwargs)

        new_ids = output_ids[0, prompt_len:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    def _generate_stream_naive(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """Naive streaming: recomputes all tokens every step (O(n^2))."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)

        generated = input_ids
        eos_id = getattr(self.tokenizer, "eos_token_id", None)

        with torch.no_grad():
            for step in range(max_tokens):
                outputs = self.model(generated)
                logits = outputs.logits[:, -1, :]

                if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)

                generated = torch.cat([generated, next_token], dim=1)

                token_str = self.tokenizer.decode(
                    next_token[0], skip_special_tokens=True
                )
                yield token_str

                if eos_id is not None and next_token.item() == eos_id:
                    break

    # ------------------------------------------------------------------
    # Fast methods — GraphableDecodeStep + StaticKVCache
    # ------------------------------------------------------------------

    def _generate_fast(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Fast generation using GraphableDecodeStep + StaticKVCache.

        Prefills the KV cache with the full prompt in one pass, then decodes
        one token at a time with constant-shape tensors (CUDA-graph safe).
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        self._static_cache.reset()

        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        tokens: list[int] = []

        with torch.no_grad():
            # Prefill: process entire prompt
            pos = torch.arange(input_ids.size(1), device=self.device).unsqueeze(0)
            logits = self._graphable(input_ids, pos)

            # First decode token from prefill logits
            next_token = self._sample_token(logits[:, -1, :], temperature)

            # Get or create persistent CUDA graph
            self._ensure_decode_graph(next_token)

            for _ in range(max_tokens):
                tok_id = next_token.item()
                tokens.append(tok_id)
                if eos_id is not None and tok_id == eos_id:
                    break

                if self._decode_graph is not None:
                    nt = next_token if next_token.dim() == 2 else next_token.unsqueeze(0)
                    self._decode_static_tok.copy_(nt)
                    self._decode_static_pos.fill_(self._static_cache.seq_len)
                    self._decode_graph.replay()
                    next_token = self._sample_token(self._decode_static_logits[:, -1, :], temperature)
                else:
                    pos = torch.tensor([[self._static_cache.seq_len]], device=self.device)
                    logits = self._graphable(next_token.unsqueeze(0) if next_token.dim() == 1 else next_token, pos)
                    next_token = self._sample_token(logits[:, -1, :], temperature)

        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _generate_stream_fast(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """Fast streaming using GraphableDecodeStep + StaticKVCache.

        After prefill, attempts to capture a CUDA graph for the decode step.
        If capture succeeds, decodes via graph replay (zero CPU overhead).
        Falls back to eager if capture fails.
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        self._static_cache.reset()

        eos_id = getattr(self.tokenizer, "eos_token_id", None)

        with torch.no_grad():
            # Prefill
            pos = torch.arange(input_ids.size(1), device=self.device).unsqueeze(0)
            logits = self._graphable(input_ids, pos)
            next_token = self._sample_token(logits[:, -1, :], temperature)

            # Get or create persistent CUDA graph
            self._ensure_decode_graph(next_token)

            for _ in range(max_tokens):
                tok_id = next_token.item()
                if eos_id is not None and tok_id == eos_id:
                    break

                token_str = self.tokenizer.decode([tok_id], skip_special_tokens=True)
                yield token_str

                if self._decode_graph is not None:
                    nt = next_token if next_token.dim() == 2 else next_token.unsqueeze(0)
                    self._decode_static_tok.copy_(nt)
                    self._decode_static_pos.fill_(self._static_cache.seq_len)
                    self._decode_graph.replay()
                    next_token = self._sample_token(self._decode_static_logits[:, -1, :], temperature)
                else:
                    pos = torch.tensor([[self._static_cache.seq_len]], device=self.device)
                    nt = next_token.unsqueeze(0) if next_token.dim() == 1 else next_token
                    logits = self._graphable(nt, pos)
                    next_token = self._sample_token(logits[:, -1, :], temperature)

    @staticmethod
    def _sample_token(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Sample or argmax a single token from logits (1, vocab_size)."""
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        return logits.argmax(dim=-1, keepdim=True)
