# Tree of Knowledge — Training Economics

> Architecture and design rationale: [tree-of-knowledge-design.md](tree-of-knowledge-design.md)

## GPU rental landscape (March 2026)

| Category | $/GPU-hr (H100) | Example |
|----------|----------------|---------|
| Marketplace (peer-to-peer) | $1.00–2.00 | vast.ai, RunPod Community |
| Mid-tier (managed) | $2.10–3.00 | Lambda, FluidStack |
| Hyperscaler (enterprise) | $3.00–6.00+ | AWS, Google Cloud, Azure |

Marketplace hosts compete on sunk costs (depreciated gaming/mining hardware), not efficiency. This creates an ideal market for ToK expert training: short, independent, fault-tolerant jobs on cheap hardware.

## Why ToK maps perfectly to marketplace GPUs

1. **No interconnect needed** — experts train independently with frozen core
2. **Short jobs** — one expert = 30-60 min on RTX 4090 (24GB sufficient)
3. **Fault tolerant** — if a host dies, restart that one expert elsewhere
4. **Small payloads** — core (~2.6GB) + expert weights (~100MB) + region data
5. **Any GPU works** — no minimum VRAM beyond fitting the core + 1 expert

## Cost comparison: 64-expert model

| Phase | Hardware | Time | Cost |
|-------|----------|------|------|
| Core training | 1× H100 | 2h | ~$5 |
| 64 experts (parallel) | 64× RTX 4090 @ $0.40/hr | 1h | ~$26 |
| Router fine-tune | 1× RTX 4090 | 15min | ~$0.10 |
| **Total** | | | **~$31** |

Traditional MoE (8× H100 NVLink, 4h): ~$80

## Scaling properties

| Experts | Parallel GPUs | Core cost (fixed) | Expert cost | Router cost | Total |
|---------|--------------|-------------------|-------------|-------------|-------|
| 64 | 64 | $5 | $26 | $0.10 | $31 |
| 256 | 256 | $5 | $103 | $0.20 | $108 |
| 1024 | 1024 | $5 | $410 | $0.50 | $416 |
| 1024 (4 waves) | 256 | $5 | $410 | $0.50 | $416 |

Expert cost scales linearly with count but wall-clock time is constant (parallel). Core cost is fixed. Router cost is negligible.

## The "SETI@home" model

Community-contributed experts:
- Anyone with a GPU downloads the frozen core (2.6GB)
- Trains an expert on their domain of interest (medical, legal, code, Dutch, etc.)
- Uploads the trained expert (~100MB)
- Community maintains a registry of available experts
- Users compose their model by selecting experts relevant to their use case

This is "Docker for model knowledge" — the core is the OS, experts are containers.

## Energy efficiency argument

The "SSD Offloading Considered Harmful" paper argues energy per useful bit matters.
ToK training is energy-efficient because:
- Each expert trains on ONLY its domain's data (no wasted compute on irrelevant tokens)
- Marketplace hosts use otherwise-idle hardware (marginal energy cost)
- Geothermal/renewable-powered hosts on vast.ai provide near-zero-carbon training
- No gradient synchronization energy cost (no all-reduce operations)
