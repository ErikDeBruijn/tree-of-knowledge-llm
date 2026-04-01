"""Tests for tensor parallelism foundation (DeviceMap)."""

from __future__ import annotations

import pytest

from grove_server.engine.tensor_parallel import DeviceMap


class TestDeviceMapSingle:
    """Single-GPU mode: all layers on device 0."""

    def test_device_map_single(self):
        """All layers map to device 0 in single mode."""
        dm = DeviceMap(num_layers=36, num_devices=1, mode="single")
        for i in range(36):
            assert dm.device_for_layer(i) == "cuda:0"

    def test_single_not_tensor_parallel(self):
        dm = DeviceMap(num_layers=36, num_devices=1, mode="single")
        assert not dm.is_tensor_parallel()

    def test_single_with_2_devices_still_uses_one(self):
        """Even with 2 devices, single mode uses device 0."""
        dm = DeviceMap(num_layers=36, num_devices=2, mode="single")
        assert dm.device_for_layer(0) == "cuda:0"
        assert dm.device_for_layer(35) == "cuda:0"


class TestDeviceMapPipeline:
    """Pipeline parallelism: first half GPU 0, second half GPU 1."""

    def test_device_map_pipeline(self):
        """First half of layers on GPU 0, second half on GPU 1."""
        dm = DeviceMap(num_layers=36, num_devices=2, mode="pipeline")
        # First 18 layers on GPU 0
        for i in range(18):
            assert dm.device_for_layer(i) == "cuda:0", f"Layer {i}"
        # Last 18 layers on GPU 1
        for i in range(18, 36):
            assert dm.device_for_layer(i) == "cuda:1", f"Layer {i}"

    def test_pipeline_odd_layers(self):
        """Odd number of layers: extra layer goes to last device."""
        dm = DeviceMap(num_layers=7, num_devices=2, mode="pipeline")
        # Floor division: 7//2 = 3 on GPU 0, rest on GPU 1
        for i in range(3):
            assert dm.device_for_layer(i) == "cuda:0"
        for i in range(3, 7):
            assert dm.device_for_layer(i) == "cuda:1"

    def test_pipeline_not_tensor_parallel(self):
        dm = DeviceMap(num_layers=36, num_devices=2, mode="pipeline")
        assert not dm.is_tensor_parallel()

    def test_pipeline_devices_list(self):
        dm = DeviceMap(num_layers=36, num_devices=2, mode="pipeline")
        assert dm.devices == ["cuda:0", "cuda:1"]


class TestDeviceMapTensor:
    """Tensor parallelism: each layer split across all devices."""

    def test_device_map_tensor(self):
        """All layers return all devices in tensor parallel mode."""
        dm = DeviceMap(num_layers=36, num_devices=2, mode="tensor")
        for i in range(36):
            devs = dm.devices_for_layer(i)
            assert devs == ["cuda:0", "cuda:1"]

    def test_tensor_is_tensor_parallel(self):
        dm = DeviceMap(num_layers=36, num_devices=2, mode="tensor")
        assert dm.is_tensor_parallel()

    def test_tensor_device_for_layer_returns_primary(self):
        """device_for_layer returns primary device in tensor mode."""
        dm = DeviceMap(num_layers=36, num_devices=2, mode="tensor")
        assert dm.device_for_layer(0) == "cuda:0"


class TestDeviceMapEdgeCases:
    """Edge cases and validation."""

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            DeviceMap(num_layers=36, num_devices=2, mode="invalid")

    def test_zero_layers_raises(self):
        with pytest.raises(ValueError):
            DeviceMap(num_layers=0, num_devices=1, mode="single")

    def test_zero_devices_raises(self):
        with pytest.raises(ValueError):
            DeviceMap(num_layers=36, num_devices=0, mode="single")

    def test_layer_out_of_range(self):
        dm = DeviceMap(num_layers=4, num_devices=1, mode="single")
        with pytest.raises(IndexError):
            dm.device_for_layer(4)
        with pytest.raises(IndexError):
            dm.device_for_layer(-1)

    def test_repr(self):
        dm = DeviceMap(num_layers=36, num_devices=2, mode="pipeline")
        r = repr(dm)
        assert "pipeline" in r
        assert "36" in r
