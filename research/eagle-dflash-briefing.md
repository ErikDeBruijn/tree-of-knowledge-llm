# Briefing: EAGLE-3 Draft Head → DFlash Upgrade Path

## Context uit parallel onderzoek

In een aparte sessie is diepgaand onderzoek gedaan naar drie non-autoregressive speculative decoding papers: DART, DFlash, en FastEagle. De conclusie versterkt jullie EAGLE-3 voorstel en geeft een duidelijke upgrade-path.

## Conclusie: Begin EAGLE-3, upgrade naar DFlash

### Stap 1 (nu): Simpele EAGLE-3 draft head
Precies zoals voorgesteld: 1 laag, ~33M params, hidden states van het base model als input, hergebruik LM head. Bewijs dat het werkt, meet acceptance rate.

### Stap 2 (upgrade): DFlash-style architectuur
DFlash (Block Diffusion, arxiv 2602.06036) is de sterkste kandidaat voor upgrade:

**Waarom DFlash:**
- **Bidirectionele attention binnen draft-blokken** — tokens in het blok zien elkaar, niet alleen voorgaande tokens. Produceert coherentere draft sequenties.
- **KV injection** — hidden states van het target model worden geïnjecteerd in de Key/Value projecties van *elke* draft laag, niet alleen als input. Dit is waarom DFlash's acceptance rate zo veel hoger is (tau=6.5 vs EAGLE-3 tau=3.0).
- **6.1× speedup** op Qwen3-8B (vs EAGLE-3 ~4-5×), **4.5× op reasoning models**.
- **Lossless** — standaard rejection sampling.
- **SGLang productie-ready**, benchmarks bij concurrency 1-32.

**Concrete architectuur:**
```
Target model (frozen):
  → extract hidden states uit ~5 uniform verdeelde lagen
  → fuse naar context features

Draft adapter (trainbaar, ~5 lagen):
  → elke laag krijgt target features via KV injection
  → bidirectionele attention BINNEN het draft blok
  → causale attention naar de prefix
  → block size 8-16 tokens
  → hergebruik target model's LM head → logits
```

**Connectie met bestaande infra:**
- De KV injection is structureel vergelijkbaar met jullie bridges — het injecteert target-model informatie in een lichtere representatie.
- De 5 draft lagen zouden rank-64 bridges kunnen hergebruiken als startpunt.
- De draft head heeft geen eigen adapters nodig — het leest hidden states die al adapter-informed zijn (het base model draait met actieve adapters voor de prefix). De competentie (bijv. Ruby) zit impliciet in de hidden states die de draft head als input krijgt.

### Stap 3 (unieke bijdrage): Gate-informed adaptive block size
Niemand anders heeft een geleerd per-token signaal dat de drafting-strategie aanpast:
- Gate laag (generieke tekst) → groot draft blok (12-16 tokens), hoge acceptance verwacht
- Gate hoog (domein-specifiek) → klein draft blok (4-8 tokens), snel verifiëren

Dit is jullie unieke contributie bovenop bestaande speculative decoding.

## Resultaten vergelijking

| Method | Speedup | Acceptance (tau) | Draft latency | Coherentie |
|--------|---------|-----------------|---------------|------------|
| EAGLE-3 | 4-5× | ~3.0 | ~10ms | Autoregressive |
| DART | 2-3.4× | ~3.5 | **1.5ms** | Laag (onafhankelijk) |
| FastEagle | 4.5-5.7× | ~5.8 | ~2ms | Medium (cascade) |
| **DFlash** | **4.9-6.1×** | **~6.5** | ~3ms | **Hoog (bidirectioneel)** |

## Voor de paper

Voeg bij de speculative decoding sectie als comment toe:

```latex
% INSIGHT FROM PARALLEL SESSION: Deep analysis of DART (2601.19278),
% DFlash (2602.06036), and FastEagle (2509.20416) shows DFlash's
% bidirectional block attention + KV injection achieves tau=6.5
% (vs EAGLE-3 tau=3.0) with 6.1x speedup. DFlash's KV injection
% architecture maps directly to our bridge surrogates. The DeltaGate
% as adaptive block size selector (large blocks when gate low,
% small blocks when gate high) is a novel contribution not present
% in any of these papers. Recommended upgrade path: EAGLE-3 head
% first (validate), then DFlash-style with gate-informed block sizing.
```

## Referenties
- DFlash: https://arxiv.org/abs/2602.06036
- DART: https://arxiv.org/abs/2601.19278
- FastEagle: https://arxiv.org/abs/2509.20416
- EAGLE-3: https://arxiv.org/abs/2503.01840
