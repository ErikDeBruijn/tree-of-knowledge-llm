"""TrainingEngine: trains LoRA adapters + delta-gates on a shared model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from grove_server.models.expert import Expert
from grove_server.models.expert_loader import DeltaGate, MoEMlpAdapter


@dataclass
class TrainingConfig:
    adapter_rank: int = 16
    adapter_alpha: int = 32  # 2 * rank
    expert_start_layer: int = 1  # Layer 1 for capability tasks
    lr_a: float = 1e-4  # LoRA+ A matrix LR
    lr_b: float = 1.6e-3  # LoRA+ B matrix LR (16x A)
    gate_lr: float = 1e-3
    gate_type: str = "contrastive"  # "contrastive" or "lm_loss"
    dropout: float = 0.1
    gate_bias_init: float = -2.0
    phase1_steps: int = 1000
    phase2_steps: int = 1500
    max_seq_len: int = 512


class TrainingEngine:
    """Train LoRA adapters and delta-gates on a shared base model.

    The model is NOT owned by this engine — it is received from the daemon
    and shared with InferenceEngine.  Training hooks are installed on the
    model's MLP layers only while training is active.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: TrainingConfig,
        device: str,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        num_layers = model.config.num_hidden_layers
        hidden_dim = model.config.hidden_size
        intermediate_dim = model.config.intermediate_size

        # Create adapter + gate modules for each covered layer
        self.adapters: dict[int, MoEMlpAdapter] = {}
        self.gates: dict[int, DeltaGate] = {}

        self._alpha_scaling = config.adapter_alpha / config.adapter_rank

        for l in range(config.expert_start_layer, num_layers):
            adapter = MoEMlpAdapter(hidden_dim, intermediate_dim, config.adapter_rank)
            # LoRA init: small random A, zero B
            nn.init.normal_(adapter.gate_lora_A, std=0.01)
            nn.init.normal_(adapter.up_lora_A, std=0.01)
            nn.init.zeros_(adapter.gate_lora_B)
            nn.init.zeros_(adapter.up_lora_B)
            # Match model dtype (BF16 on GPU)
            model_dtype = next(model.parameters()).dtype
            self.adapters[l] = adapter.to(device=device, dtype=model_dtype)

            gate = DeltaGate(hidden_dim)
            nn.init.zeros_(gate.linear.weight)
            nn.init.constant_(gate.linear.bias, config.gate_bias_init)
            model_dtype = next(model.parameters()).dtype
            self.gates[l] = gate.to(device=device, dtype=model_dtype)

        self.step: int = 0
        self.phase: int = 1  # 1=adapter, 2=gate, 3=contrastive gate
        self._hooks_installed: bool = False
        self._original_mlps: dict[int, nn.Module] = {}
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._best_checkpoint: Optional[dict] = None
        self._best_quality: float = 0.0
        self._fp8_step = None  # Set by daemon if FP8 graphable available
        self._setup_optimizer()

    def _setup_optimizer(self) -> None:
        """Create optimizer for the current phase with LoRA+ differential LR."""
        if self.phase == 1:
            # LoRA+: separate A (low LR) and B (high LR) param groups
            a_params = []
            b_params = []
            for adapter in self.adapters.values():
                a_params.extend([adapter.gate_lora_A, adapter.up_lora_A])
                b_params.extend([adapter.gate_lora_B, adapter.up_lora_B])
            self._optimizer = torch.optim.AdamW([
                {"params": a_params, "lr": self.config.lr_a},
                {"params": b_params, "lr": self.config.lr_b},
            ], weight_decay=0.05)
        else:
            params = []
            for gate in self.gates.values():
                params.extend(gate.parameters())
            self._optimizer = torch.optim.AdamW(params, lr=self.config.gate_lr)

    def install_hooks(self) -> None:
        """Install training hooks on model MLP layers."""
        if self._hooks_installed:
            return
        layers = self.model.model.layers
        for l in self.adapters:
            if l >= len(layers):
                break
            layer = layers[l]
            self._original_mlps[l] = layer.mlp
            layer.mlp = _TrainingMLP(
                original_mlp=self._original_mlps[l],
                adapter=self.adapters[l],
                gate=self.gates[l] if self.phase >= 2 else None,
                alpha_scaling=self._alpha_scaling,
                fp8_step=self._fp8_step,
                layer_idx=l,
            )
        self._hooks_installed = True

    def uninstall_hooks(self) -> None:
        """Restore original model MLP layers."""
        if not self._hooks_installed:
            return
        layers = self.model.model.layers
        for l, original_mlp in self._original_mlps.items():
            if l < len(layers):
                layers[l].mlp = original_mlp
        self._original_mlps.clear()
        self._hooks_installed = False

    def train_step(self, input_ids: torch.Tensor) -> dict:
        """One adapter training step (phase 1). Returns metrics dict."""
        input_ids = input_ids.to(self.device)
        if input_ids.size(1) < 2:
            return {"loss": 0.0, "step": self.step, "phase": self.phase}

        for adapter in self.adapters.values():
            adapter.train()

        out = self.model(input_ids)
        loss = F.cross_entropy(
            out.logits[:, :-1].reshape(-1, out.logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
        )

        self._optimizer.zero_grad()
        loss.backward()
        params = []
        for a in self.adapters.values():
            params.extend(a.parameters())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        self._optimizer.step()

        self.step += 1
        return {"loss": loss.item(), "step": self.step, "phase": self.phase}

    def contrastive_gate_step(
        self, domain_ids: torch.Tensor, generic_ids: torch.Tensor
    ) -> dict:
        """One contrastive gate training step (phase 2).

        Trains gates with direct discriminative signal:
        L_gate = -log(gate(domain)) - log(1 - gate(generic))

        Also includes a small LM loss to preserve generation quality.
        """
        domain_ids = domain_ids.to(self.device)
        generic_ids = generic_ids.to(self.device)
        if domain_ids.size(1) < 2 or generic_ids.size(1) < 2:
            return {"loss": 0.0, "step": self.step, "phase": self.phase, "selectivity": 0.0}

        for gate in self.gates.values():
            gate.train()

        HS = self.model.config.hidden_size
        start = self.config.expert_start_layer
        num_layers = self.model.config.num_hidden_layers

        # Get hidden states for domain and generic
        with torch.no_grad():
            d_out = self.model(domain_ids, output_hidden_states=True)
            g_out = self.model(generic_ids, output_hidden_states=True)

        # Contrastive loss: push domain UP, generic DOWN
        contrastive_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        domain_gate_sum = 0.0
        generic_gate_sum = 0.0
        n_layers = 0

        for l in range(start, num_layers):
            if l not in self.gates:
                continue
            d_hs = d_out.hidden_states[l].reshape(-1, HS).detach()
            g_hs = g_out.hidden_states[l].reshape(-1, HS).detach()

            d_gate = torch.sigmoid(self.gates[l](d_hs))
            g_gate = torch.sigmoid(self.gates[l](g_hs))

            contrastive_loss = contrastive_loss + (
                -torch.log(d_gate + 1e-8).mean()
                - torch.log(1 - g_gate + 1e-8).mean()
            )
            domain_gate_sum += d_gate.mean().item()
            generic_gate_sum += g_gate.mean().item()
            n_layers += 1

        # Small LM loss to preserve generation quality
        lm_loss = F.cross_entropy(
            self.model(domain_ids).logits[:, :-1].reshape(-1, self.model.config.vocab_size),
            domain_ids[:, 1:].reshape(-1),
        )

        total_loss = 0.1 * contrastive_loss / max(n_layers, 1) + lm_loss

        self._optimizer.zero_grad()
        total_loss.backward()
        params = [p for g in self.gates.values() for p in g.parameters()]
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        self._optimizer.step()

        selectivity = (domain_gate_sum - generic_gate_sum) / max(n_layers, 1)
        self.step += 1
        return {
            "loss": total_loss.item(),
            "contrastive_loss": contrastive_loss.item() / max(n_layers, 1),
            "lm_loss": lm_loss.item(),
            "selectivity": selectivity,
            "domain_gate": domain_gate_sum / max(n_layers, 1),
            "generic_gate": generic_gate_sum / max(n_layers, 1),
            "step": self.step,
            "phase": self.phase,
        }

    def checkpoint_if_better(self, quality: float) -> bool:
        """Save internal checkpoint if quality improves. Returns True if saved."""
        if quality > self._best_quality:
            self._best_quality = quality
            self._best_checkpoint = self.state_dict()
            return True
        return False

    def restore_best(self) -> None:
        """Restore the best checkpoint."""
        if self._best_checkpoint is not None:
            self.load_state_dict(self._best_checkpoint)

    def switch_phase(self, phase: int) -> None:
        """Switch training phase. Phase 2 freezes adapters, trains gates."""
        was_hooked = self._hooks_installed
        if was_hooked:
            self.uninstall_hooks()

        self.phase = phase
        if phase == 2:
            # Freeze adapters
            for adapter in self.adapters.values():
                for p in adapter.parameters():
                    p.requires_grad_(False)
                adapter.eval()
            # Unfreeze gates
            for gate in self.gates.values():
                for p in gate.parameters():
                    p.requires_grad_(True)

        self._setup_optimizer()
        if was_hooked:
            self.install_hooks()

    def to_expert(self, name: str) -> Expert:
        """Export current state as an Expert for inference."""
        # Deep-copy adapters and gates so training can continue independently
        adapters: dict[int, nn.Module] = {}
        gates: dict[int, nn.Module] = {}
        for l in self.adapters:
            adapters[l] = _clone_module(self.adapters[l])
            gates[l] = _clone_module(self.gates[l])

        start_layer = self.config.expert_start_layer
        end_layer = self.model.config.num_hidden_layers
        return Expert(
            name=name,
            start_layer=start_layer,
            end_layer=end_layer,
            skip_layers=set(),
            bridge_layers=set(),
            adapters=adapters,
            gates=gates,
            bridges={},
        )

    @property
    def is_active(self) -> bool:
        """Whether training hooks are installed."""
        return self._hooks_installed

    def state_dict(self) -> dict:
        """For checkpointing."""
        return {
            "adapters": {l: a.state_dict() for l, a in self.adapters.items()},
            "gates": {l: g.state_dict() for l, g in self.gates.items()},
            "step": self.step,
            "phase": self.phase,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from checkpoint."""
        for l, adapter_state in state["adapters"].items():
            self.adapters[l].load_state_dict(adapter_state)
        for l, gate_state in state["gates"].items():
            self.gates[l].load_state_dict(gate_state)
        self.step = state["step"]
        self.phase = state["phase"]
        if self.phase == 2:
            self.switch_phase(2)
        self._setup_optimizer()


class _TrainingMLP(nn.Module):
    """Wraps original MLP with adapter (and optionally gate) for training.

    Works with both BF16 (standard nn.Linear) and FP8 (weight=None,
    uses graphable._fp8_linear instead). The fp8_step + layer_idx
    enable FP8-compatible training without dequantizing weights.
    """

    def __init__(
        self,
        original_mlp: nn.Module,
        adapter: MoEMlpAdapter,
        gate: Optional[DeltaGate] = None,
        alpha_scaling: float = 1.0,
        fp8_step=None,
        layer_idx: int = -1,
    ):
        super().__init__()
        self.original_mlp = original_mlp
        self.adapter = adapter
        self.gate = gate
        self.alpha_scaling = alpha_scaling
        self._fp8_step = fp8_step
        self._layer_idx = layer_idx
        # Expose sub-modules for attribute access
        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj

    def _proj(self, proj_name: str, x: torch.Tensor) -> torch.Tensor:
        """Run a projection, using FP8 matmul if weights were quantized."""
        proj = getattr(self.original_mlp, proj_name)
        if hasattr(proj, 'weight') and proj.weight is not None:
            return proj(x)
        # Weight is None → FP8 quantized. Use graphable step's FP8 matmul.
        if self._fp8_step is not None:
            key = f"{self._layer_idx}.mlp.{proj_name}"
            flat = x.reshape(-1, x.size(-1))
            out = self._fp8_step._fp8_linear(flat, key)
            return out.reshape(*x.shape[:-1], -1)
        raise RuntimeError(f"{proj_name}.weight is None and no FP8 step available")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_out = self._proj("gate_proj", x)
        up_out = self._proj("up_proj", x)

        # Alpha scaling on adapter corrections (LoRA+ best practice)
        gate_corr = self.adapter.gate_correction(x) * self.alpha_scaling
        up_corr = self.adapter.up_correction(x) * self.alpha_scaling

        activated = F.silu(gate_out + gate_corr) * (up_out + up_corr)
        adapted = self._proj("down_proj", activated.reshape(-1, activated.size(-1)))
        adapted = adapted.reshape(*x.shape[:-1], -1)

        if self.gate is not None:
            base_gate = self._proj("gate_proj", x)
            base_up = self._proj("up_proj", x)
            base_act = F.silu(base_gate) * base_up
            base_out = self._proj("down_proj", base_act.reshape(-1, base_act.size(-1)))
            base_out = base_out.reshape(*x.shape[:-1], -1)
            gate_value = torch.sigmoid(self.gate(x))
            return base_out + gate_value * (adapted - base_out)

        return adapted


def _clone_module(module: nn.Module) -> nn.Module:
    """Create a detached deep copy of an nn.Module."""
    import copy
    clone = copy.deepcopy(module)
    for p in clone.parameters():
        p.requires_grad_(False)
    return clone
