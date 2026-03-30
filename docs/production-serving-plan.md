# Production Serving Plan (geparkeerd)

## Huidige situatie
- PyTorch inference: 26 tok/s (top-3 sparse, 10 adapters)
- Bottleneck: Python dispatch overhead (~7 tok/s per adapter), niet compute
- Rank-16 adapters zijn 0.03B params — verwaarloosbaar naast 8B trunk

## Aanbevolen pad: vLLM + custom routing proxy
1. vLLM met S-LoRA/Punica kernels voor batched multi-LoRA
2. Routing proxy doet gate logit forward pass, selecteert top-3 adapters
3. vLLM doet de zware inference met geselecteerde LoRA's

## Verwachte performance
- vLLM met multi-LoRA: ~50-60 tok/s (dicht bij base)
- GGUF quantisatie (Q4): nog sneller, vergelijkbaar met llama.cpp

## Top-k benchmark referentie (PyTorch, 10 adapters)
| Mode | tok/s | Skip rate |
|------|-------|-----------|
| Base | 60.8 | 100% |
| Top-1 | 38.2 | 90% |
| Top-2 | 31.2 | 80% |
| Top-3 | 26.2 | 70% |
| Dense | 12.5 | 0% |

## Niet nu — eerst paper review + research consolidatie
