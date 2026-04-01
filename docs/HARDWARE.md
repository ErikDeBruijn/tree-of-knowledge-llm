# Hardware Specifications & Multi-GPU Inference Analysis

Workstation configuration for Tree-of-Knowledge LLM research.

## Component Specifications

| Component | Model | Key Specs |
|-----------|-------|-----------|
| **CPU** | Intel Core Ultra 7 265K (Arrow Lake-S) | 20 cores (8P+12E), 5.5 GHz boost, LGA 1851 |
| **Motherboard** | MSI MEG Z890 ACE | Z890 chipset, ATX, Wi-Fi 7, Thunderbolt 4 |
| **GPU (x2)** | NVIDIA RTX PRO 6000 Blackwell Max-Q | 96 GB GDDR7 ECC, 300W TDP each |
| **RAM** | Crucial Pro 128 GB (2x64 GB) DDR5-5600 | CL46, dual-channel |

## GPU Specifications (RTX PRO 6000 Blackwell Max-Q)

| Spec | Value |
|------|-------|
| CUDA Cores | 24,064 |
| Tensor Cores | 752 (5th gen) |
| RT Cores | 188 (4th gen) |
| VRAM | 96 GB GDDR7 ECC (nvidia-smi reports ~102 GB total*) |
| Memory Bus | 512-bit |
| Memory Bandwidth | 1,792 GB/s |
| PCIe Interface | PCIe 5.0 x16 |
| TDP | 300W (Max-Q) / 600W (Workstation Edition) |
| FP32 Peak | 110 TFLOPS |
| FP4 AI Peak | 3,511 TOPS (with sparsity) |
| NVLink | **Not supported** |
| Multi-Instance GPU | Up to 4 MIG instances @ 24 GB each |

\* nvidia-smi may report ~102,400 MiB rather than the marketed 96 GB (98,304 MiB).
The extra ~4 GB is likely GDDR7 spare capacity on the 512-bit bus (12 chips x 8.5 GB
per chip = ~102 GB physical). With ECC disabled, more usable VRAM may be exposed.

## CPU PCIe Lane Budget

The Core Ultra 7 265K provides:
- **20x PCIe 5.0 lanes** from the CPU (16 for GPU + 4 for NVMe)
- **4x PCIe 4.0 lanes** from the CPU (for a second NVMe)
- **DMI 4.0 x8** link to Z890 chipset (which provides additional PCIe 4.0 lanes)

The 16 PCIe 5.0 GPU lanes are shared between PCI_E1 and PCI_E2. This is a consumer
platform -- there are only 16 PCIe 5.0 lanes for graphics, not 32.

## PCIe Topology & Lane Sharing

### MSI MEG Z890 ACE Slot Configuration

| Slot | Max Config | Source | Physical Size |
|------|-----------|--------|---------------|
| PCI_E1 | PCIe 5.0 x16 | CPU | x16 |
| PCI_E2 | PCIe 5.0 x8 | CPU | x16 (physical) |
| PCI_E3 | PCIe 4.0 x4 | Z890 chipset | x16 (physical) |

### CPU PCIe Lane Configuration Table (from MSI manual, page 33)

PCI_E1, PCI_E2, and M2_4 share the same 20 CPU PCIe 5.0 lanes:

| PCI_E1 | PCI_E2 | M2_4 | Notes |
|--------|--------|------|-------|
| 5.0 x16 | -- | N/A | Single GPU, full bandwidth |
| 5.0 x8+x8 | -- | N/A | PCI_E1 bifurcated, PCI_E2 unused |
| 5.0 x8+x4+x4 | -- | N/A | PCI_E1 bifurcated 3-way |
| 5.0 x8 | 5.0 x8 | N/A | **Two GPUs, each x8** |
| 5.0 x8 | 5.0 x4+x4 | N/A | PCI_E2 bifurcated |
| 5.0 x8 | 5.0 x4 | 5.0 x4 | GPU + GPU(x4) + NVMe Gen5 |

Key: `--` = no signal, `N/A` = M2_4 uses chipset PCIe 4.0 x4 by default.

### Text-based PCIe Topology Diagram

```
                    Intel Core Ultra 7 265K
                    ========================
                    |                      |
             PCIe 5.0 x16*           PCIe 5.0 x4        PCIe 4.0 x4
             (shared pool)           (dedicated)         (dedicated)
                    |                      |                   |
            +-------+-------+          M2_1 slot          M2_2 slot
            |               |          (NVMe)             (NVMe)
        PCI_E1          PCI_E2
     (up to x16)     (up to x8)
            |               |
    RTX PRO 6000    RTX PRO 6000
     GPU #0 (x8)     GPU #1 (x8)        DMI 4.0 x8
                                             |
                                      Z890 Chipset
                                      ===========
                                      |    |    |
                                   PCI_E3  M2_3  M2_5
                                  (4.0 x4)(4.0x4)(4.0x4)
                                             |
                                          M2_4**

  *  With 2 GPUs installed: PCI_E1=x8, PCI_E2=x8 (mandatory split)
  ** M2_4 defaults to chipset PCIe 4.0 x4, but CAN use CPU 5.0 x4
     (this further reduces PCI_E2 to x4 — avoid this with dual GPUs)
```

### GPU Interconnect

nvidia-smi topo shows **PHB** (PCIe Host Bridge) between the two GPUs. This means:
- Both GPUs connect to the CPU's PCIe root complex
- Inter-GPU communication goes: GPU0 -> PCIe switch -> CPU -> PCIe switch -> GPU1
- No NVLink, no direct peer-to-peer bridge

## Bandwidth Analysis

### Per-GPU PCIe Bandwidth

| Configuration | Per-direction BW | Bidirectional BW |
|--------------|-----------------|------------------|
| PCIe 5.0 x16 (single GPU) | 63.0 GB/s | 126.0 GB/s |
| PCIe 5.0 x8 (dual GPU) | 31.5 GB/s | 63.0 GB/s |
| PCIe 1.0 x8 (current!) | 2.0 GB/s | 4.0 GB/s |

### Current Problem

nvidia-smi reports **PCIe Gen1 x8**. This is catastrophically wrong:

| Metric | Gen1 x8 (current) | Gen5 x8 (correct) | Ratio |
|--------|-------------------|-------------------|-------|
| Bandwidth | 2.0 GB/s | 31.5 GB/s | **15.75x** |

At Gen1 x8, the PCIe link is the dominant bottleneck for any operation that touches
CPU-GPU data transfer: model loading, KV-cache spilling, tensor-parallel all-reduce,
and even initial weight loading.

### Bandwidth Context for Inference

| Pathway | Bandwidth |
|---------|-----------|
| GPU VRAM (per GPU) | 1,792 GB/s |
| PCIe 5.0 x8 (per GPU, corrected) | 31.5 GB/s |
| PCIe 5.0 x16 (single GPU) | 63.0 GB/s |
| DDR5-5600 dual-channel (CPU RAM) | ~89.6 GB/s |
| Inter-GPU via PCIe (round-trip) | ~31.5 GB/s effective |

For single-GPU inference, PCIe bandwidth barely matters -- the model lives entirely
in VRAM and the memory-bound decode loop runs at 1,792 GB/s.

For tensor-parallel across 2 GPUs, the all-reduce communication must traverse PCIe.
At Gen5 x8, the ~31.5 GB/s inter-GPU bandwidth is ~57x slower than VRAM bandwidth.
This creates a significant overhead for each transformer layer's all-reduce step.

## Expected Inference Performance

### Single GPU (96 GB VRAM, models that fit)

Based on published benchmarks for RTX PRO 6000 (full 600W edition, ~5-14% faster
than Max-Q):

| Model | Quantization | Expected tok/s (Max-Q) |
|-------|-------------|----------------------|
| Llama 3.3 70B | Q4 | ~28-30 |
| DeepSeek-R1 70B | Q4 | ~28-30 |
| Qwen 2.5 72B | Q4 | ~25-28 |
| GPT-OSS 120B (MoE, 5.1B active) | Q4 | ~115-130 |
| DeepSeek-R1 32B | Q4 | ~55-60 |
| Gemma 3 27B | Q4 | ~53-58 |

### Dual GPU Tensor Parallel (192 GB total VRAM)

With tensor parallelism over PCIe (no NVLink), expect:
- **70B models**: ~1.3-1.5x single-GPU speed (not 2x, due to PCIe communication overhead)
- **120B+ dense models at higher quant**: enabled by memory capacity, ~20-40 tok/s
- **Very large models (200B+)**: possible with 192 GB combined VRAM, but PCIe x8 will be the bottleneck

The main value of the second GPU is **memory capacity** (192 GB total), not throughput
scaling. For models that fit in 96 GB, a single GPU will often be faster than tensor
parallel across two GPUs over PCIe.

### Pipeline Parallelism Alternative

For dual-GPU setups without NVLink, pipeline parallelism (splitting layers between
GPUs) can be more efficient than tensor parallelism for batch-size-1 interactive use:
- Lower communication overhead (only activations at the split point)
- Each GPU processes its layers at full VRAM bandwidth
- Latency per token increases (sequential processing), but less PCIe traffic

## BIOS Fix: PCIe Gen1 x8 -> Gen5 x8

The GPUs are running at PCIe Gen1 x8, which needs to be fixed immediately.

### BIOS Menu Path

**Advanced Mode (F7) -> Settings -> Advanced -> PCIe Sub-system Settings**

In this submenu:

1. **PCI_E1 Gen Mode** -- change from `[Auto]` to `[Gen5]`
2. **PCI_E2 Gen Mode** -- change from `[Auto]` to `[Gen5]`
3. **CPU PCIe Lanes Configuration** -- ensure it is set to `x8/x8` (for dual GPU)
4. **Re-Size BAR Support** -- set to `[Enabled]`

### Additional BIOS Settings to Check

| Setting | Location | Recommended |
|---------|----------|-------------|
| Above 4G Decoding | PCIe Sub-system Settings (implicit with ReBAR) | Enabled |
| Re-Size BAR Support | PCIe Sub-system Settings | Enabled |
| PCI_E1 Gen Mode | PCIe Sub-system Settings | Gen5 |
| PCI_E2 Gen Mode | PCIe Sub-system Settings | Gen5 |
| CPU PCIe Lanes Config | PCIe Sub-system Settings | x8/x8 for dual GPU |
| ASPM (PEG 1/2) | PCIe Sub-system Settings | Disabled (for max performance) |

### Why Auto Might Default to Gen1

Possible causes for Gen1 fallback:
- BIOS defaults to Safe Boot mode with lower PCIe speeds
- The Smart Button "Safe Boot" feature boots with "default and lower PCIe (from CPU) mode"
- A BIOS update may have reset PCIe settings to conservative defaults
- PCIe link training failure causing fallback (check physical seating of GPUs)

### Verification After BIOS Change

After applying settings, verify with:
```bash
# Check link speed and width
nvidia-smi -q | grep -A5 "PCI"

# Expected output should show:
#   Link Gen: 5
#   Link Width: 8x

# Also verify with lspci
sudo lspci -vvs $(lspci | grep NVIDIA | head -1 | cut -d' ' -f1) | grep -i "lnksta"
# Should show: Speed 32GT/s (Gen5), Width x8
```

## Recommendations Summary

1. **Immediate**: Fix PCIe Gen Mode in BIOS from Auto to Gen5 for both slots.
   This alone will provide a **15.75x bandwidth improvement**.

2. **For models up to ~90 GB (Q4 70B, Q8 32B, etc.)**: Use a single GPU.
   Single-GPU inference avoids all PCIe overhead and is simpler to configure.

3. **For models requiring >96 GB VRAM**: Use pipeline parallelism (layer split)
   rather than tensor parallelism when running interactive (batch=1) inference.
   Tensor parallel is better for batched throughput workloads.

4. **Enable Resizable BAR**: Small but measurable improvement for GPU memory access
   patterns, especially during model loading.

5. **Avoid using M2_4 in PCIe 5.0 mode**: This would steal x4 lanes from PCI_E2,
   reducing it from x8 to x4 -- a 50% bandwidth cut to the second GPU.

6. **Platform limitation to accept**: This is a consumer Z890 platform with 20 PCIe
   5.0 lanes. True dual x16 requires HEDT/workstation platforms (Threadripper PRO,
   Xeon W). The x8/x8 split is the best this platform can do and is adequate for
   most inference workloads where VRAM bandwidth (1,792 GB/s) dominates.
