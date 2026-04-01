"""Tensor and pipeline parallelism foundation for multi-GPU inference.

Provides DeviceMap: a layer-to-device mapping abstraction that supports
three modes of parallelism. This is the foundation — actual tensor-parallel
matmul splitting is not yet implemented.

Modes:
- 'single': all layers on one GPU (current default)
- 'pipeline': first half on GPU 0, second half on GPU 1
- 'tensor': each layer split across both GPUs (requires custom matmul)
"""

from __future__ import annotations

_VALID_MODES = ("single", "pipeline", "tensor")


class DeviceMap:
    """Maps model layers to devices for tensor/pipeline parallelism.

    This class encodes the strategy for distributing model layers across
    available GPUs. It does not move tensors — it provides the mapping
    that other components (GraphableDecodeStep, StaticKVCache) use to
    decide where to place and execute each layer.
    """

    def __init__(
        self,
        num_layers: int,
        num_devices: int,
        mode: str = "single",
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {_VALID_MODES}, got '{mode}'"
            )
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")
        if num_devices <= 0:
            raise ValueError(f"num_devices must be > 0, got {num_devices}")

        self.num_layers = num_layers
        self.num_devices = num_devices
        self.mode = mode

        # Build the device list
        self.devices: list[str] = [f"cuda:{i}" for i in range(num_devices)]

        # Pre-compute layer → device mapping
        if mode == "single":
            self._layer_map: list[str] = ["cuda:0"] * num_layers
        elif mode == "pipeline":
            split = num_layers // num_devices
            self._layer_map = []
            for i in range(num_layers):
                dev_idx = min(i // split, num_devices - 1) if split > 0 else 0
                self._layer_map.append(f"cuda:{dev_idx}")
        elif mode == "tensor":
            # In tensor parallel, every layer lives on all devices.
            # _layer_map stores the primary device (device 0).
            self._layer_map = ["cuda:0"] * num_layers

    def device_for_layer(self, layer_idx: int) -> str:
        """Return the primary device for a given layer.

        In tensor parallel mode, returns the primary device (cuda:0).
        Use devices_for_layer() for the full list.

        Args:
            layer_idx: Layer index (0-based).

        Returns:
            Device string like 'cuda:0' or 'cuda:1'.

        Raises:
            IndexError: If layer_idx is out of range.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {self.num_layers})"
            )
        return self._layer_map[layer_idx]

    def devices_for_layer(self, layer_idx: int) -> list[str]:
        """Return all devices a layer is split across.

        For single/pipeline mode, returns a single-element list.
        For tensor mode, returns all devices.

        Args:
            layer_idx: Layer index (0-based).

        Returns:
            List of device strings.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {self.num_layers})"
            )
        if self.mode == "tensor":
            return list(self.devices)
        return [self._layer_map[layer_idx]]

    def is_tensor_parallel(self) -> bool:
        """Whether this mapping uses tensor parallelism."""
        return self.mode == "tensor"

    def __repr__(self) -> str:
        return (
            f"DeviceMap(num_layers={self.num_layers}, "
            f"num_devices={self.num_devices}, mode='{self.mode}')"
        )
