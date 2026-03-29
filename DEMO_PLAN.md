# Forest of Knowledge — Demo Plan

## Doel
Een live demo die niet te negeren is. Laat zien dat het base model
specifieke kennis mist, en dat een kilobyte-grote adapter die kennis
toevoegt — in milliseconden, hot-pluggable.

## Demo-flow
1. Stel het base model een niche-vraag
2. Base model faalt (hallucineert of zegt "ik weet het niet")
3. Laad de adapter (toon bestandsgrootte: 64 KB)
4. Stel dezelfde vraag
5. Model antwoordt correct en gedetailleerd
6. Verwijder de adapter, model faalt weer
7. Laad een ANDERE adapter, ander domein verbetert

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
