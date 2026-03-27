# ToK Runner — Execute Experiments

Execute on ollama.local (ssh root@ollama.local). Use `/root/t6b-venv/bin/python3`.

Read `paper/mogae-paper-v4.tex` Section 4 (Experiments) for context on what
has been run and what the current architecture looks like.

## Pre-flight checklist

1. Check for rogue processes and GPU availability
2. Verify checkpoint exists before loading
3. Create checkpoint directory before training
4. Fix known bugs preemptively (see below)

## Known bugs (fix in every new script)

```python
# 1. Always detach before numpy
tensor.detach().float().cpu().numpy()  # NOT .cpu().numpy()

# 2. Hook unwrapping at phase transitions
if hasattr(base_mlp, '_orig_mlp'):
    base_mlp = base_mlp._orig_mlp
hook._orig_mlp = base_mlp  # store after creating hook

# 3. GQA RoPE: expand before reshape
cos = cos.expand(B, -1, -1)  # before .reshape(B * T, ...)
```

## Every experiment must include

- M_ij ablation matrix on final checkpoint (the PRIMARY metric)
- Routing selectivity per domain
- PPL on eval set
