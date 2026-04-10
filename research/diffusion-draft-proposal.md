# Diffusion LLM als Intuïtief Draft Model voor Autoregressive Verificatie

## Claim (Speculative)

Een diffusion LLM kan dienen als "intuïtief" draft model dat coherente gedachtegangen in één keer produceert, geverifieerd door een autoregressive LLM die token-level precisie levert. De diffusion draft is niet een goedkope benadering van het echte model — het heeft een complementaire kracht: het vormt gedachten in samenhang (bidirectioneel) waar het autoregressive model sequentieel en fragiel is.

**Confidence class: Speculative.** Geen experimentele validatie. Alle claims hieronder zijn theoretisch.

## Kernidee

### Observatie 1: De residual stream bevat meer dan concrete tokens
De hidden state in een transformer bevat op elk punt niet alleen informatie over het huidige token, maar ook vage "richtingen" — potentiële concepten, nog niet geconcretiseerd. Dit is geobserveerd in mechanistic interpretability onderzoek: middenlagen houden meerdere mogelijke continuaties tegelijk open. Pas late lagen convergeren naar één token.

**Evidence status:** Supported (Anthropic mechanistic interpretability, meerdere replicaties).

### Observatie 2: Autoregressive generatie is fragiel voor redenering
Bij chain-of-thought generatie moet het model bij elk token een onomkeerbare keuze maken. Eén verkeerde token kan een hele redenering laten ontsporen. Het model kan niet "terugkijken" en corrigeren.

**Evidence status:** Observed (breed gerapporteerd in CoT literature).

### Observatie 3: Diffusion modellen vormen de hele output tegelijk
Een diffusion LLM begint met ruis en verfijnt iteratief naar een coherente sequentie. Het gebruikt bidirectionele attention — elk token ziet alle andere tokens. Dit is niet "vooruit kijken naar conclusies die nog niet bestaan." Het is: alle tokens tegelijk naar coherentie duwen.

**Evidence status:** Observed (diffusion LLM papers, MDLM, SEDD, etc.).

### De synthese (Speculative)
De "intuïties" in de residual stream — vage richtingen, potentiële concepten — zijn precies wat een diffusion model nodig heeft. Het hoeft geen concrete tokens te voorspellen; het vormt een coherent geheel waarin die vage richtingen tegelijk tot expressie komen. Het autoregressive model is beter in per-token precisie maar slechter in globale coherentie.

Combinatie:
- **Diffusion draft**: produceert een coherente gedachtegang (meerdere tokens) in één shot. Snel. Mogelijk beter in globale coherentie dan autoregressive generatie.
- **Autoregressive verifier**: checkt token voor token of de distributie correct is. Corrigeert waar nodig via rejection sampling. Precies.

## Hypotheses

### H1: Gedeelde architectuur is mogelijk
Een diffusion head en een autoregressive head kunnen hetzelfde base model delen (Q/K/V projecties, FFN lagen). Alleen de attention mask (causaal vs bidirectioneel) en de output head verschillen.

- **What would falsify this:** Als de geleerde representaties in een autoregressive model fundamenteel incompatibel zijn met bidirectionele attention — als de FFN weights informatie coderen die alleen werkt met causale masking.
- **Alternative explanation:** Het zou kunnen dat gedeelde weights een compromis opleveren dat voor beide taken suboptimaal is.

### H2: Diffusion draft produceert coherentere CoT dan autoregressive draft
Bij reasoning-taken (multi-step, keten van afhankelijkheden) produceert een diffusion draft sequenties die globaal coherenter zijn dan autoregressive drafts van vergelijkbare grootte.

- **What would falsify this:** Als bidirectionele attention geen voordeel biedt voor CoT coherentie — als de autoregressive factorizatie geen informatie verliest die relevant is voor globale samenhang.
- **How to measure:** Percentage van gegenereerde CoT-sequenties dat tot het correcte antwoord leidt, draft vs draft, vóór verificatie.

### H3: De grove-gate voorspelt waar diffusion draft faalt
Hoge gate-activatie (domein-specifieke kennis nodig) correleert met lagere acceptance rate van diffusion draft tokens.

- **What would falsify this:** Als de gate geen voorspellende waarde heeft voor diffusion-autoregressive agreement.
- **Depends on:** H1 (gedeelde architectuur moet eerst werken).

### H4: Bridges en layer skipping werken wél voor diffusion draft
Optimalisaties die gefalsificeerd zijn voor autoregressive generatie (compounding error) zijn wél bruikbaar in een diffusion draft, omdat er geen autoregressive loop is — fouten compounderen niet.

- **What would falsify this:** Als de iteratieve refinement stappen van diffusion hun eigen vorm van compounding error hebben.
- **Evidence needed:** Vergelijk diffusion draft kwaliteit met/zonder bridges en layer skipping.

## Open vragen (onbeantwoord)

1. **Rejection sampling math.** Standaard speculative decoding gebruikt conditionele distributies P(token_n | token_1..n-1). Een diffusion model produceert een joint distributie P(token_1..n). Kan rejection sampling aangepast worden voor joint-naar-conditionele verificatie? Dit is niet triviaal en mogelijk een fundamenteel obstakel.

2. **Hoeveel diffusion stappen zijn nodig?** Als het 10 iteraties kost om een coherente sequentie te produceren, en elke iteratie een forward pass is, dan is de draft niet snel. De snelheidswinst hangt af van: (diffusion_stappen × forward_pass_kosten) vs (k × autoregressive_forward_pass_kosten).

3. **Training van de diffusion head.** Hoe train je een diffusion head bovenop een frozen autoregressive base model? De base is getraind met causale masking — werken de geleerde representaties met bidirectionele attention?

4. **Welke tokens profteren?** De hypothese is dat "boilerplate" tokens (lage gate, voorspelbaar) goed werken met diffusion en "reasoning" tokens (hoge gate, domein-specifiek) autoregressive verificatie nodig hebben. Maar dit is een aanname, geen observatie.

## Wat we NIET weten (honest uncertainty)

- Of de rejection sampling math überhaupt werkbaar is voor diffusion→autoregressive verificatie
- Of een gedeeld base model met twee heads competitief is met twee gespecialiseerde modellen
- Of diffusion LLMs voor tekst überhaupt voldoende kwaliteit leveren (het veld is jong)
- Of de snelheidswinst reëel is na alle overhead (diffusion iteraties, verificatie, KV cache management)

## Relatie tot Grove of Knowledge

Dit voorstel is complementair aan de grove-architectuur maar niet afhankelijk ervan. De grove-specifieke voordelen:
- Gate als selector tussen diffusion draft (lage gate) en autoregressive (hoge gate)
- Bridges en layer skipping hergebruikt voor de diffusion path
- Adapters alleen op de autoregressive verifier (domeinkennis hoeft niet in de draft)

## Aanbevolen volgorde van onderzoek

1. **Literatuurstudie:** Bestaande diffusion LLM papers (MDLM, SEDD, Mercury) op kwaliteit en snelheid. Is de basis solide genoeg?
2. **Rejection sampling math:** Kan joint→conditionele verificatie? Pen en papier eerst, niet implementeren.
3. **Gedeelde architectuur test:** Voeg een diffusion head toe aan een frozen autoregressive model. Meet of de gedeelde representaties bruikbaar zijn.
4. **Kwaliteitsvergelijking:** Diffusion draft vs autoregressive draft op CoT-taken. Is er een coherentievoordeel?
5. **Pas dan:** integratie met grove, bridges, gate-informed routing.

**Stop-criterium na stap 2:** Als de rejection sampling math niet werkbaar is zonder fundamentele aanpassingen, is het hele voorstel onhaalbaar in de voorgestelde vorm.

## Confidence samenvatting

| Claim | Confidence |
|-------|-----------|
| Residual stream bevat abstracte "richtingen" | Supported |
| Autoregressive CoT is fragiel | Observed |
| Diffusion vormt output tegelijk | Observed |
| Diffusion draft is coherenter voor CoT | Speculative |
| Gedeelde architectuur is mogelijk | Speculative |
| Rejection sampling is aanpasbaar | Speculative |
| Grove-gate voorspelt diffusion acceptance | Speculative |
| Snelheidswinst is reëel | Speculative |
