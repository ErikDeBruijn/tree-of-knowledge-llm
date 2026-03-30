"""Shared adapter module definitions for the Grove of Knowledge.

All training, validation, and composition scripts import from here
instead of copy-pasting class definitions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_EXPERT_START = 12
DEFAULT_RANK = 16
DEFAULT_BIAS_INIT = -2.0


class LoRA(nn.Module):
    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        self.A = nn.Parameter(torch.randn(in_dim, rank, dtype=torch.bfloat16) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, out_dim, dtype=torch.bfloat16))

    def forward(self, x):
        return x @ self.A @ self.B


class Expert(nn.Module):
    def __init__(self, hidden_size, intermediate_size, rank):
        super().__init__()
        self.gate_lora = LoRA(hidden_size, intermediate_size, rank)
        self.up_lora = LoRA(hidden_size, intermediate_size, rank)

    def forward(self, x, base_mlp):
        return base_mlp.down_proj(
            F.silu(base_mlp.gate_proj(x) + self.gate_lora(x))
            * (base_mlp.up_proj(x) + self.up_lora(x))
        )


class DeltaGate(nn.Module):
    """Per-layer scalar gate on the adapter delta.

    Returns raw logit (not sigmoid) — caller applies activation.
    For single-adapter: use sigmoid.
    For multi-adapter: use softmax across adapters.
    """
    def __init__(self, hidden_size, bias_init=DEFAULT_BIAS_INIT):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1, bias=True, dtype=torch.bfloat16)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, bias_init)

    def forward(self, x):
        return self.linear(x)

    def gate_sigmoid(self, x):
        return torch.sigmoid(self.forward(x))


class HookModule(nn.Module):
    """Wraps a function as an nn.Module for replacing model.layers[l].mlp."""
    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def forward(self, x):
        return self._fn(x)


def create_adapter_and_gates(hidden_size, intermediate_size, n_layers, rank,
                             expert_start, bias_init=DEFAULT_BIAS_INIT, device="cpu"):
    """Create adapter experts and gates for layers expert_start..n_layers-1."""
    adapter = nn.ModuleDict()
    gates = nn.ModuleDict()
    for l in range(expert_start, n_layers):
        adapter[str(l)] = Expert(hidden_size, intermediate_size, rank).to(device)
        gates[str(l)] = DeltaGate(hidden_size, bias_init).to(device)
    return adapter, gates


def load_adapter_package(path, hidden_size, intermediate_size, n_layers, device="cpu"):
    """Load adapter + gates from a standardized package directory."""
    import os, json
    adapter_pt = os.path.join(path, "adapter.pt")
    manifest_path = os.path.join(path, "manifest.json")

    ckpt = torch.load(adapter_pt, map_location="cpu", weights_only=False)
    rank = ckpt.get("rank", DEFAULT_RANK)
    expert_start = ckpt.get("expert_start", DEFAULT_EXPERT_START)

    adapter, gates = create_adapter_and_gates(
        hidden_size, intermediate_size, n_layers, rank, expert_start, device=device
    )
    adapter.load_state_dict(ckpt["adapter"])
    gates.load_state_dict(ckpt["gates"])
    adapter.eval()
    gates.eval()
    for p in adapter.parameters():
        p.requires_grad_(False)
    for p in gates.parameters():
        p.requires_grad_(False)

    manifest = None
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    return adapter, gates, ckpt, manifest
