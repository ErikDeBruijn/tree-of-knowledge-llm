# Grove of Knowledge — Demo Plan

## Doel
Een live demo die niet te negeren is. Drie wow-momenten:

1. **"I don't know, yet"** — het model herkent dat het de kennis niet
   heeft (via learntropy) en STOPT met hallucineren
2. **Hot-plug knowledge** — een 64KB adapter wordt geladen en het model
   antwoordt correct ("I know Kung Fu")
3. **Semantic discovery** — het model vindt zelf de juiste adapter op
   basis van de query-embedding

## Kernbijdrage: hallucinatie-reductie
Huidige modellen hallucineren omdat ze geen mechanisme hebben om te
detecteren dat ze iets NIET weten. Learntropy biedt dat mechanisme:
- Hoge learntropy op een query = het model is onzeker
- Geen actieve adapter die learntropy verlaagt = "I don't know"
- Adapter beschikbaar die learntropy WEL verlaagt = "I don't know yet,
  want to download the ✓ verified knowledge module?"
- Na laden: learntropy daalt, antwoord is correct

Dit verandert hallucinatie van een onoplosbaar probleem naar een
routing-probleem: stuur de query naar de juiste adapter, of geef toe
dat je het niet weet.

## Demo-flow ("I know Kung Fu")

### Scenario 1: Proactieve adapter discovery
```
User: "What are the key principles of Wing Chun?"

Model: "I don't have specialized knowledge about Wing Chun.
        However, I found a ✓ verified knowledge module:
        'Martial Arts: Wing Chun' (64 KB).
        Would you like me to load it?"

User: "Yes"

[Loading adapter... 12ms]

Model: "I know Kung Fu.

Wing Chun is a concept-based Chinese martial art with these
core principles:
1. Centerline theory: protect and attack along the centerline
2. Simultaneous defense and attack (Lin Siu Dai Da)
3. Economy of movement..."
```

### Scenario 2: Hot-swap demonstratie
```
[Met Wing Chun adapter geladen]
User: "What about the latest EU AI Act regulations?"
Model: "I don't have that specialized knowledge loaded.
        Found: '✓ EU AI Act 2026' (128 KB). Load?"
User: "Yes"
[Swap: unload Wing Chun, load EU AI Act... 15ms]
Model: "The EU AI Act entered full force in February 2026..."
```

### Scenario 3: Hallucinatie vs eerlijkheid
```
[Zonder adapter]
User: "What does Article 23b of the Dutch Healthcare Insurance Act say?"
Model: "I don't have specific knowledge about Dutch healthcare
        legislation. My learntropy for this topic is high (3.8),
        indicating I'm likely to hallucinate if I try to answer.
        No verified adapter found for this topic yet."

[Dit is het cruciale moment: het model STOPT in plaats van te
 hallucineren. Dat is nieuw.]
```

## Niche-topics voor adapters (wow-factor)

### Fun & wow-factor (Matrix reference)
- **Martial arts**: Wing Chun principles, specific kata sequences,
  historical lineage — "I know Kung Fu" moment
- **Mixology**: obscure cocktail recipes and techniques that even
  bartenders google — surprisingly knowledge-dense domain
- **Chess openings**: deep lines of obscure openings (Latvian Gambit,
  Budapest Gambit) with move-by-move analysis

### Impressive & verifiable
- **Medical protocols**: specific treatment protocol for a rare
  condition (e.g., step-by-step Guillain-Barré management)
- **Pharmacology**: drug interactions discovered in 2025-2026
- **EU AI Act 2026**: specific articles that entered force after
  the base model's training cutoff

### Technical & provable
- **Recent framework docs**: API documentation for a project released
  after the cutoff (e.g., a new major version of a popular library)
- **Hardware specs**: specific benchmark numbers for hardware released
  after training (verifiable against official specs)
- **New language features**: Python 3.14 or Rust 2024 edition features
  the base model doesn't know about

### Current events
- **Recent scientific discovery**: a specific paper from Nature/Science
  published after cutoff — adapter knows the finding, base doesn't
- **Space exploration**: specific mission data from 2025-2026 missions
- **Climate data**: specific 2025-2026 measurements the base can't know

## Metingen per adapter
- Bestandsgrootte (KB)
- Laadtijd (ms)
- PPL verbetering op domein-specifieke eval set
- Correctheid van antwoorden (menselijke evaluatie)
- Base model score op dezelfde vragen (0% verwacht)

## Vergelijking
Dezelfde kennis toevoegen via:
- **(a) Full fine-tuning**: hoeveel kost dat? (GPU-uren, VRAM)
- **(b) RAG**: hoeveel context tokens? Latency?
- **(c) Adapter**: 64 KB, <1ms laadtijd, nul extra tokens

## Technische vereisten
- Inference server met hot-plug API endpoint
- Adapter upload/download endpoint
- Latency logging per request
- A/B toggle in de UI (base vs adapter)
