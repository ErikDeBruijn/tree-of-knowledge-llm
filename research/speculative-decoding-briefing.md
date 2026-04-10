# Briefing: Gate-Informed Speculative Decoding Experiment

## Wat is er gewijzigd

### Paper (`paper/mogae-paper-v6.tex`)
- **Sectie "Gate as Precision-Budget Allocator"**: herschreven van packed INT4 registers naar **segmented mantissa** FP storage. Sign+exponent zijn altijd beschikbaar; alleen mantissa-segmenten worden incrementeel gelezen. Dit maakt variable precision native FP-compatibel — cruciaal omdat training en inference hetzelfde geheugenformaat delen.
- **Nieuwe sectie "Gate-Informed Speculative Decoding"**: beschrijft hoe de grove's eigen variable-precision path als draft model fungeert voor speculative decoding, met het volledige model als verifier.
- **Nieuwe referenties**: speculative decoding papers (Leviathan 2022, Chen 2023), NCA paper (Lee 2026).
- **Nieuwe paragraph "Base model selection"** bij Limitations: overweging om NCA pre-pre-training te gebruiken voor een beter base model.

### Nieuw experiment design (`research/gate-speculative-decoding-experiment.md`)
Volledig experiment plan met hypotheses, metrics, en go/no-go gates.

## Het idee

Het probleem: variable precision (FP8 draft path) + bridges geven snelheid, maar elke kleine fout compound over lange sequenties. Na 50 tokens kan het model afgedwaald zijn.

De oplossing: gebruik het model's eigen variable-precision als speculative decoding setup:
- **Draft**: FP8 (sign + exponent + mantissa segment A = 8 bits/weight) + bridges voor low-gate lagen → snel, goedkoop
- **Verifier**: FP16 (alle mantissa-segmenten = 16 bits/weight) + adapters → exact correcte distributie
- **Rejection sampling** garandeert dat de output identiek is aan het full-precision model → lossless

Wat dit uniek maakt t.o.v. standaard speculative decoding:
1. Geen apart draft model nodig — zelfde weights, andere mantissa-resolutie
2. De DeltaGate **voorspelt** waar de draft zal falen (hoge gate = adapter nodig = rejection waarschijnlijk)
3. Per-laag selectieve verificatie: alleen high-gate lagen naar FP16, rest blijft FP8
4. Compounding error is onmogelijk: rejection sampling reset de distributie

## Wat te doen

Lees `research/gate-speculative-decoding-experiment.md` voor het volledige plan. Begin met **Phase 1** (Experiments 1-2: acceptance rate meting). Dit is puur meetwerk — geen speculative decoding loop nodig, alleen meten hoe vaak FP8 en FP16 hetzelfde token produceren.

**Go/no-go**: als overall acceptance rate < 60%, is speculative decoding niet viable. Stop en rapporteer als negatief resultaat.

## Segmented mantissa — waarom niet INT4

Packed INT4 registers (eerder prototype) werkten voor inference maar niet voor training: integer-formaten missen de dynamische range voor gradient berekening. Segmented mantissa houdt sign+exponent altijd intact, waardoor elke precision level native floating point is. Training backward pass kan dezelfde variable-precision weights gebruiken als de forward pass — geen format conversie nodig.
