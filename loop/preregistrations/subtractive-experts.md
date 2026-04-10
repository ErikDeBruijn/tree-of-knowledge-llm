# Pre-registration: Subtractive Experts

## Concept

Tot nu toe zijn experts altijd additief — ze voegen capability of kennis toe.
Maar experts kunnen ook subtractief zijn: het base model "uitkleden" voor
taken die minder compute nodig hebben.

Een router beslist per token of het volledige model nodig is, of dat een
uitgeklede variant (bridges, quantized, layer-skipped) voldoende is.

## Hypotheses

### H1: Eenvoudige taken gebruiken het base model niet volledig
- **Predictie:** Voor summarization, simple QA, en parafrasering is de output
  van een uitgekleed model (bridges op 50% van de lagen) identiek aan het
  volledige model in >80% van de tokens.
- **Nul:** <60% token agreement.
- **Rationale:** Kleine modellen kunnen deze taken ook. Het verschil zit in
  de lagen die "overtollig" zijn voor eenvoudige taken.

### H2: Gate kan complexiteit voorspellen
- **Predictie:** Een getrainde gate (contrastief: complex vs simpel tekst)
  kan per-token voorspellen welke lagen nodig zijn, met Pearson r > 0.5
  tussen gate activatie en token-complexiteit (gemeten als full-model hidden
  state divergence van uitgekleed model).
- **Nul:** Geen correlatie.

### H3: Subtractieve routing geeft compute besparing zonder kwaliteitsverlies
- **Predictie:** Adaptieve routing (uitkleden voor simpele tokens, volledig
  model voor complexe) geeft >30% throughput verbetering met <2% PPL
  degradatie op gemengde workloads.
- **Nul:** <10% throughput verbetering of >5% PPL degradatie.

### H4: Subtractive + additive experts composeren
- **Predictie:** Een subtractieve "fast mode" expert en een additieve
  "Ruby code" expert kunnen simultaan actief zijn — de router gebruikt het
  uitgeklede model voor simpele Ruby en het volledige model + adapter voor
  complexe Ruby.
- **Nul:** Compositionele conflicten.

## Connectie met bestaand werk

- Layer skipping: al GEFALSIFICEERD voor alle lagen (catastrophic token loss).
  Maar bridges als surrogaat WERKEN voor vroege lagen (MSE 0.19 L13).
- Gate: al bewezen als domain detector (selectivity 0.96).
  Kan hergebruikt worden als complexiteit detector.
- Variable precision: de Triton per-group FP8 kernel IS al een subtractieve
  operatie (minder precision = minder bandwidth = sneller).

## Risico's

- "Simpel" vs "complex" is subjectiever dan "domain" vs "generic"
- Bridge kwaliteit degradeert in deep layers (MSE 12.0 L32)
- Compounding error bij te veel bridges (eerder gefalsificeerd voor skipping)

## Status: EXPERIMENT 1-2 UITGEVOERD
Datum: 2026-04-04

### Resultaten
- H1 DEELS BEVESTIGD: 31/36 lagen veilig te vervangen individueel (>70% agreement).
  Maar cumulatief 8 lagen: 65.2% agreement (onder 75% threshold).
- Speedup 1.10x (8 bridges) — beperkt omdat attention (40%) ongewijzigd.
- Compounding error: individueel >90%, maar samen 65%. Bridges zijn te
  onnauwkeurig als je er 8+ combineert.

### Volgende stappen
- Hogere rank bridges (128/256) om compounding te verminderen
- Attention ook vervangen (attention skipping of low-rank attention)
- FP8 per laag ipv bridge — geen approximatie-error
- Gate-informed selectie: vervang alleen de N meest tolerante lagen per token
