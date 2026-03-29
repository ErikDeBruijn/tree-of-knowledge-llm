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

### Actueel & grappig
- **Bizarre rechtszaken 2025-2026**: de zaak van de man die zijn buurman
  aanklaagde vanwege een te luid kakelende kip (NL jurisprudentie)
- **AI-wetgeving EU 2026**: specifieke artikelen uit de AI Act die net
  van kracht zijn geworden — het base model kent de oude versie
- **Crypto-regulering**: nieuwe MiCA-regels die na de training cutoff
  van het base model zijn ingegaan

### Indrukwekkend & nuttig
- **Medisch protocol**: een specifiek behandelprotocol voor zeldzame
  aandoening (bijv. Guillain-Barré syndroom stapsgewijze behandeling)
- **Farmacologie**: interacties tussen specifieke medicijnen die in
  2025-2026 zijn ontdekt
- **Lokale kennis**: specifieke Nederlandse gemeentelijke regelgeving
  (bijv. Utrechtse parkeernormen 2026)

### Technisch & verifieerbaar
- **Recent open-source project**: API-documentatie van een project dat
  na de cutoff is gereleased (bijv. een nieuwe versie van een framework)
- **Hardware specs**: RTX 5090 Blackwell specificaties en benchmarks
  (als die na de cutoff zijn)
- **Nieuwe programmeertaal features**: Python 3.14 features die het
  base model niet kent

### Cultureel & fun
- **Recent boek/film**: plot details van een boek/film uit 2026
- **Sportresultaten**: specifieke wedstrijduitslagen na de cutoff
- **Memes**: recente internet-cultuur die na training is ontstaan

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
