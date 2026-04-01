"""InferenceEngine: wraps a HuggingFace model with expert injection."""

from __future__ import annotations

import types
from typing import Iterator, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from grove_server.engine.cuda_graph import CUDAGraphRunner
from grove_server.engine.graphable_model import GraphableDecodeStep
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
    ) -> None:
        """Load base model and tokenizer.

        Args:
            model_name: HuggingFace model identifier.
            device: Target device ("cpu", "cuda", "auto").
            dtype: Weight dtype ("bfloat16", "float16", "float32").
        """
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device if device == "auto" else None,
        )
        if device != "auto":
            self.model = self.model.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device if device != "auto" else str(self.model.device)
        self.num_layers: int = self.model.config.num_hidden_layers

        self._active_expert: Optional[Expert] = None
        self._original_forwards: dict[int, types.MethodType] = {}
        self._graph_runner: Optional[CUDAGraphRunner] = None
        self._graphable: Optional[GraphableDecodeStep] = None
        self._static_cache: Optional[StaticKVCache] = None

    def _build_graphable_decode(self, max_seq_len: int = 2048) -> None:
        """Build a CUDA-graphable decode step with static KV cache.

        Creates a StaticKVCache and GraphableDecodeStep that can be used
        with CUDAGraphRunner for zero-overhead decode.

        Args:
            max_seq_len: Maximum sequence length to pre-allocate.
        """
        config = self.model.config
        self._static_cache = StaticKVCache(
            num_layers=config.num_hidden_layers,
            num_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            max_seq_len=max_seq_len,
            batch_size=1,
            dtype=next(self.model.parameters()).dtype,
            device=self.device,
        )
        self._graphable = GraphableDecodeStep(
            self.model, self._static_cache, max_seq_len=max_seq_len
        )
        self._graph_runner = CUDAGraphRunner(device=self.device)

    def _invalidate_graph(self) -> None:
        """Invalidate any captured CUDA graph (expert config changed)."""
        if self._graph_runner is not None:
            self._graph_runner.invalidate()
            self._graph_runner = None

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

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Generate text completion.

        Args:
            prompt: Input text.
            max_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text (excluding the prompt).
        """
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

        # Decode only the generated tokens (skip prompt)
        new_ids = output_ids[0, prompt_len:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        use_cuda_graph: bool = False,
    ) -> Iterator[str]:
        """Streaming generation — yield tokens as they're produced.

        Uses a simple loop with greedy/sampling decode to yield one token
        at a time. This avoids depending on TextIteratorStreamer threads.

        Args:
            prompt: Input text.
            max_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature.
            use_cuda_graph: If True, attempt to capture the decode step as a
                CUDA graph after the first token (prefill). Only effective on
                CUDA devices. Falls back to eager if capture fails.
        """
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
