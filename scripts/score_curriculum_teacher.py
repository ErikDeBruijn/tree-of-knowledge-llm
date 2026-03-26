#!/usr/bin/env python3
"""
Score curriculum chunks with teacher model (Qwen3-30B-A3B) to compute ZPD scores.

Uses llama-cpp-python to load the GGUF model and compute per-chunk perplexity.
Compares teacher PPL against student PPL to identify Zone of Proximal Development.

ZPD concept: chunks where student struggles but teacher doesn't (high student_ppl / teacher_ppl ratio)
indicate material that IS learnable but the student hasn't mastered yet.
"""

import json
import math
import time
import sys
import random
import numpy as np
from pathlib import Path
from scipy import stats

# --- Configuration ---
MODEL_PATH = Path.home() / ".ollama/models/blobs/sha256-58574f2e94b99fb9e4391408b57e5aeaaaec10f6384e9a699fc2cb43a5c8eabf"
STUDENT_DATA = Path("/tmp/curriculum_scored.json")
RESULTS_DIR = Path.home() / "github.com/erikdebruijn/tree-of-knowledge-llm/results"
SAMPLE_SIZE = 500
CONTEXT_SIZE = 2048  # enough for 512-token chunks
SEED = 42
ZPD_THRESHOLD = 2.0  # student_ppl / teacher_ppl > this = ZPD candidate

def compute_perplexity(model, text: str) -> tuple[float, int]:
    """Compute perplexity of text using the model.

    Returns (perplexity, token_count).
    """
    # Tokenize
    tokens = model.tokenize(text.encode("utf-8"), add_bos=True)
    n_tokens = len(tokens)

    if n_tokens < 2:
        return float('nan'), n_tokens

    # Evaluate all tokens at once
    model.reset()
    model.eval(tokens)

    # Collect log-likelihoods from the logits
    # For each position i, we want log P(token[i] | token[0..i-1])
    # model.eval gives us logits for all positions

    # Actually, llama-cpp-python's eval works differently.
    # We need to use the scores (logits) that are stored after eval.
    # But the Python binding only exposes scores for the last eval call.

    # Better approach: process tokens one context-window at a time
    # and accumulate log-likelihoods.

    total_nll = 0.0
    n_scored = 0

    # Reset and process
    model.reset()

    # Process in batches but we need per-token logits
    # Use model._eval to get all logits
    batch_size = min(n_tokens, CONTEXT_SIZE)
    batch = tokens[:batch_size]

    # Evaluate the batch
    model.reset()
    model.eval(batch)

    # Now get logits for each position
    # In llama-cpp-python, after eval, we can access scores
    # scores shape: (n_tokens, n_vocab)

    # The scores buffer contains logits for the last eval call
    scores = model.scores  # numpy array of shape (n_tokens_evaluated, vocab_size)

    for i in range(1, len(batch)):
        # logits at position i-1 predict token at position i
        logits = scores[i - 1]
        # Apply log-softmax
        max_logit = np.max(logits)
        log_sum_exp = max_logit + np.log(np.sum(np.exp(logits - max_logit)))
        log_prob = logits[batch[i]] - log_sum_exp
        total_nll -= log_prob
        n_scored += 1

    if n_scored == 0:
        return float('nan'), n_tokens

    avg_nll = total_nll / n_scored
    ppl = math.exp(avg_nll)
    return ppl, n_tokens


def main():
    random.seed(SEED)

    # Load student data
    print(f"Loading student data from {STUDENT_DATA}...")
    with open(STUDENT_DATA) as f:
        all_chunks = json.load(f)
    print(f"  Total chunks: {len(all_chunks)}")

    # Sample
    if len(all_chunks) > SAMPLE_SIZE:
        # Stratified sample: take proportionally from each difficulty quartile
        sorted_chunks = sorted(all_chunks, key=lambda x: x.get('ppl', 0))
        quartile_size = len(sorted_chunks) // 4
        sample = []
        per_quartile = SAMPLE_SIZE // 4
        for q in range(4):
            start = q * quartile_size
            end = start + quartile_size if q < 3 else len(sorted_chunks)
            quartile = sorted_chunks[start:end]
            sample.extend(random.sample(quartile, min(per_quartile, len(quartile))))
        # Fill remainder if needed
        remaining = SAMPLE_SIZE - len(sample)
        if remaining > 0:
            pool = [c for c in all_chunks if c not in sample]
            sample.extend(random.sample(pool, min(remaining, len(pool))))
        chunks = sample[:SAMPLE_SIZE]
    else:
        chunks = all_chunks

    print(f"  Sampled {len(chunks)} chunks (stratified by student PPL)")

    # Load teacher model
    print(f"\nLoading teacher model from {MODEL_PATH}...")
    print("  (This will take ~30-60 seconds and ~18GB RAM)")

    from llama_cpp import Llama

    t0 = time.time()
    model = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=CONTEXT_SIZE,
        n_batch=512,
        n_gpu_layers=-1,  # offload all to Metal
        logits_all=True,  # we need logits for all positions, not just last
        verbose=False,
        seed=SEED,
    )
    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s")

    # Score chunks
    print(f"\nScoring {len(chunks)} chunks...")
    results = []
    start_time = time.time()
    errors = 0

    for i, chunk in enumerate(chunks):
        text = chunk.get('text', '')
        student_ppl = chunk.get('ppl', float('nan'))

        try:
            teacher_ppl, n_tokens = compute_perplexity(model, text)
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  Error on chunk {i}: {e}")
            teacher_ppl = float('nan')
            n_tokens = 0

        if not math.isnan(teacher_ppl) and not math.isnan(student_ppl) and teacher_ppl > 0:
            zpd_score = student_ppl / teacher_ppl
        else:
            zpd_score = float('nan')

        results.append({
            'text_preview': text[:100],
            'student_ppl': student_ppl,
            'teacher_ppl': teacher_ppl,
            'zpd_score': zpd_score,
            'token_count': n_tokens,
            'difficulty_percentile': chunk.get('difficulty_percentile', None),
        })

        # Progress
        if (i + 1) % 25 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            eta = (len(chunks) - i - 1) / (rate / 60) if rate > 0 else 0
            last_zpd = f"{zpd_score:.2f}" if not math.isnan(zpd_score) else "NaN"
            print(f"  [{i+1}/{len(chunks)}] {rate:.1f} chunks/min, "
                  f"ETA {eta:.0f}s, last ZPD={last_zpd}, errors={errors}")

    total_time = time.time() - start_time

    # --- Analysis ---
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")

    # Filter valid results
    valid = [r for r in results if not math.isnan(r['teacher_ppl']) and not math.isnan(r['student_ppl'])]
    print(f"\nValid results: {len(valid)} / {len(results)} ({errors} errors)")
    print(f"Throughput: {len(chunks) / total_time * 60:.1f} chunks/min ({total_time:.0f}s total)")

    if len(valid) < 10:
        print("ERROR: Too few valid results for analysis")
        sys.exit(1)

    student_ppls = np.array([r['student_ppl'] for r in valid])
    teacher_ppls = np.array([r['teacher_ppl'] for r in valid])
    zpd_scores = np.array([r['zpd_score'] for r in valid])

    # 1. Correlation
    spearman_r, spearman_p = stats.spearmanr(student_ppls, teacher_ppls)
    pearson_r, pearson_p = stats.pearsonr(student_ppls, teacher_ppls)

    print(f"\n--- Correlation ---")
    print(f"Spearman rho: {spearman_r:.4f} (p={spearman_p:.2e})")
    print(f"Pearson r:    {pearson_r:.4f} (p={pearson_p:.2e})")

    # Pre-registered prediction check
    if spearman_r < 0.7:
        print(f"  >>> PASS: rho={spearman_r:.3f} < 0.7 — teacher and student disagree on difficulty")
    elif spearman_r > 0.9:
        print(f"  >>> FAIL: rho={spearman_r:.3f} > 0.9 — ZPD is likely empty")
    else:
        print(f"  >>> AMBIGUOUS: rho={spearman_r:.3f} in [0.7, 0.9] range")

    # 2. ZPD fraction
    zpd_fraction = np.mean(zpd_scores > ZPD_THRESHOLD)
    print(f"\n--- ZPD Analysis ---")
    print(f"ZPD fraction (ZPD > {ZPD_THRESHOLD}): {zpd_fraction:.1%} ({np.sum(zpd_scores > ZPD_THRESHOLD)}/{len(valid)})")

    if 0.15 <= zpd_fraction <= 0.40:
        print(f"  >>> PASS: ZPD fraction {zpd_fraction:.1%} in [15%, 40%] range")
    else:
        print(f"  >>> OUT OF RANGE: ZPD fraction {zpd_fraction:.1%} (expected 15-40%)")

    # 3. Distribution stats
    print(f"\n--- PPL Distributions ---")
    print(f"Student PPL:  mean={np.mean(student_ppls):.2f}, median={np.median(student_ppls):.2f}, "
          f"std={np.std(student_ppls):.2f}")
    print(f"Teacher PPL:  mean={np.mean(teacher_ppls):.2f}, median={np.median(teacher_ppls):.2f}, "
          f"std={np.std(teacher_ppls):.2f}")
    print(f"ZPD score:    mean={np.mean(zpd_scores):.2f}, median={np.median(zpd_scores):.2f}, "
          f"std={np.std(zpd_scores):.2f}")

    # 4. Percentile breakdown
    print(f"\n--- ZPD by Student Difficulty Quartile ---")
    for q in range(4):
        q_low = np.percentile(student_ppls, q * 25)
        q_high = np.percentile(student_ppls, (q + 1) * 25)
        mask = (student_ppls >= q_low) & (student_ppls <= q_high)
        q_zpd = zpd_scores[mask]
        if len(q_zpd) > 0:
            q_zpd_frac = np.mean(q_zpd > ZPD_THRESHOLD)
            print(f"  Q{q+1} (student PPL {q_low:.1f}-{q_high:.1f}): "
                  f"ZPD fraction={q_zpd_frac:.1%}, mean ZPD={np.mean(q_zpd):.2f}, n={len(q_zpd)}")

    # 5. Example high-ZPD chunks
    print(f"\n--- Top 10 High-ZPD Chunks ---")
    sorted_valid = sorted(valid, key=lambda r: r['zpd_score'], reverse=True)
    for i, r in enumerate(sorted_valid[:10]):
        print(f"  {i+1}. ZPD={r['zpd_score']:.2f} | student={r['student_ppl']:.1f} | "
              f"teacher={r['teacher_ppl']:.1f} | '{r['text_preview'][:60]}...'")

    # 6. Example low-ZPD chunks (both find easy)
    print(f"\n--- Top 5 Low-ZPD Chunks (both easy) ---")
    sorted_low = sorted(valid, key=lambda r: r['zpd_score'])
    for i, r in enumerate(sorted_low[:5]):
        print(f"  {i+1}. ZPD={r['zpd_score']:.2f} | student={r['student_ppl']:.1f} | "
              f"teacher={r['teacher_ppl']:.1f} | '{r['text_preview'][:60]}...'")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "teacher_scoring_500.json"

    output = {
        'metadata': {
            'teacher_model': 'qwen3:30b-a3b',
            'student_model': 'Qwen3-0.6B (from curriculum_scored.json)',
            'sample_size': len(chunks),
            'valid_results': len(valid),
            'errors': errors,
            'total_time_seconds': total_time,
            'chunks_per_minute': len(chunks) / total_time * 60,
            'seed': SEED,
            'zpd_threshold': ZPD_THRESHOLD,
            'context_size': CONTEXT_SIZE,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        },
        'summary': {
            'spearman_rho': float(spearman_r),
            'spearman_p': float(spearman_p),
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'zpd_fraction': float(zpd_fraction),
            'student_ppl_mean': float(np.mean(student_ppls)),
            'student_ppl_median': float(np.median(student_ppls)),
            'teacher_ppl_mean': float(np.mean(teacher_ppls)),
            'teacher_ppl_median': float(np.median(teacher_ppls)),
            'zpd_mean': float(np.mean(zpd_scores)),
            'zpd_median': float(np.median(zpd_scores)),
        },
        'pre_registered': {
            'spearman_rho_pass': float(spearman_r) < 0.7,
            'zpd_fraction_in_range': 0.15 <= float(zpd_fraction) <= 0.40,
        },
        'results': results,
    }

    # Convert NaN to None for JSON serialization
    def sanitize(obj):
        if isinstance(obj, float) and math.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(sanitize(output), f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Verdict
    print(f"\n{'='*60}")
    if spearman_r < 0.7 and 0.15 <= zpd_fraction <= 0.40:
        print("VERDICT: ZPD concept VALIDATED")
        print("Teacher and student meaningfully disagree on chunk difficulty.")
        print("A significant fraction of chunks are in the Zone of Proximal Development.")
    elif spearman_r > 0.9:
        print("VERDICT: ZPD concept REJECTED")
        print("Teacher and student largely agree — ZPD is empty.")
    else:
        print("VERDICT: MIXED — needs further investigation")
        print(f"Spearman rho={spearman_r:.3f}, ZPD fraction={zpd_fraction:.1%}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
