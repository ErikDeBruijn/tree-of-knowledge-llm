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
    expert_start_layer: int = 12
    lr: float = 3e-4
    gate_lr: float = 1e-3
    l1_lambda: float = 0.05
    gate_bias_init: float = -2.0
    phase1_steps: int = 500
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

        for l in range(config.expert_start_layer, num_layers):
            adapter = MoEMlpAdapter(hidden_dim, intermediate_dim, config.adapter_rank)
            # Kaiming init for A matrices, zero for B (standard LoRA init)
            nn.init.kaiming_uniform_(adapter.gate_lora_A)
            nn.init.kaiming_uniform_(adapter.up_lora_A)
            self.adapters[l] = adapter.to(device)

            gate = DeltaGate(hidden_dim)
            nn.init.zeros_(gate.linear.weight)
            nn.init.constant_(gate.linear.bias, config.gate_bias_init)
            self.gates[l] = gate.to(device)

        self.step: int = 0
        self.phase: int = 1
        self._hooks_installed: bool = False
        self._original_mlps: dict[int, nn.Module] = {}
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._setup_optimizer()

    def _setup_optimizer(self) -> None:
        """Create optimizer for the current phase."""
        if self.phase == 1:
            params = []
            for adapter in self.adapters.values():
                params.extend(adapter.parameters())
            self._optimizer = torch.optim.AdamW(params, lr=self.config.lr)
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
                gate=self.gates[l] if self.phase == 2 else None,
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
        """One training step. Returns {"loss": float, "step": int, "phase": int}."""
        input_ids = input_ids.to(self.device)
        if input_ids.size(1) < 2:
            return {"loss": 0.0, "step": self.step, "phase": self.phase}

        # Set trainable modules to train mode
        for adapter in self.adapters.values():
            adapter.train()
        for gate in self.gates.values():
            gate.train()

        # Forward
        out = self.model(input_ids)
        logits = out.logits

        # Language modeling loss
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
        )

        # L1 sparsity on gate biases in phase 2
        if self.phase == 2:
            for gate in self.gates.values():
                gate_out = gate(torch.zeros(1, self.model.config.hidden_size, device=self.device))
                loss = loss + self.config.l1_lambda * gate_out.mean()

        self._optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        if self.phase == 1:
            params = []
            for a in self.adapters.values():
                params.extend(a.parameters())
            torch.nn.utils.clip_grad_norm_(params, 1.0)
        else:
            params = []
            for g in self.gates.values():
                params.extend(g.parameters())
            torch.nn.utils.clip_grad_norm_(params, 1.0)
        self._optimizer.step()

        self.step += 1
        return {"loss": loss.item(), "step": self.step, "phase": self.phase}

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
    """Wraps original MLP with adapter (and optionally gate) for training."""

    def __init__(
        self,
        original_mlp: nn.Module,
        adapter: MoEMlpAdapter,
        gate: Optional[DeltaGate] = None,
    ):
        super().__init__()
        self.original_mlp = original_mlp
        self.adapter = adapter
        self.gate = gate
        # Expose sub-modules so the model's attribute access still works
        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_out = self.original_mlp.gate_proj(x)
        up_out = self.original_mlp.up_proj(x)

        adapted = self.original_mlp.down_proj(
            F.silu(gate_out + self.adapter.gate_correction(x))
            * (up_out + self.adapter.up_correction(x))
        )

        if self.gate is not None:
            # Phase 2: gate blends base and adapted output
            base_out = self.original_mlp(x)
            gate_value = self.gate(x)
            return base_out + gate_value * (adapted - base_out)

        return adapted


def _clone_module(module: nn.Module) -> nn.Module:
    """Create a detached deep copy of an nn.Module."""
    import copy
    clone = copy.deepcopy(module)
    for p in clone.parameters():
        p.requires_grad_(False)
    return clone
