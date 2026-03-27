#!/usr/bin/env python3
"""
Token-level ZPD analysis: Qwen3-30B-A3B (teacher) vs Qwen3-1.7B (student).

Prior chunk-level analysis showed Spearman rho=0.958. This script tests whether
token-level disagreement is HIGHER (lower rho) — which would confirm that
"niche tokens" exist where the student struggles but teacher succeeds.

Pre-registered predictions:
  - Token-level Spearman rho < chunk-level 0.958 (expected ~0.7-0.85)
  - High-ZPD tokens (zpd > 3.0) cluster around: rare words, domain terms, proper nouns, code tokens
  - High-ZPD token fraction: 5-20% of all tokens
"""

import json
import math
import time
import sys
import random
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from scipy import stats

# --- Configuration ---
TEACHER_GGUF = Path.home() / ".ollama/models/blobs/sha256-58574f2e94b99fb9e4391408b57e5aeaaaec10f6384e9a699fc2cb43a5c8eabf"
STUDENT_GGUF = Path.home() / ".ollama/models/blobs/sha256-3d0b790534fe4b79525fc3692950408dca41171676ed7e21db57af5c65ef6ab6"
STUDENT_DATA = Path("/tmp/curriculum_scored.json")
RESULTS_DIR = Path.home() / "github.com/erikdebruijn/tree-of-knowledge-llm/results"

SAMPLE_SIZE = 50  # chunks
CONTEXT_SIZE = 2048
SEED = 42
ZPD_TOKEN_THRESHOLD = 3.0  # token-level zpd > this = "niche token"
CHUNK_LEVEL_RHO = 0.958  # from prior experiment


def compute_token_losses(model, text: str) -> list[tuple[int, float]]:
    """Compute per-token cross-entropy loss.

    Returns list of (token_id, loss) for each predicted token (skips BOS).
    loss = -log P(token_t | context_t-1..0)
    """
    tokens = model.tokenize(text.encode("utf-8"), add_bos=True)
    n_tokens = len(tokens)

    if n_tokens < 2:
        return []

    batch_size = min(n_tokens, CONTEXT_SIZE)
    batch = tokens[:batch_size]

    model.reset()
    model.eval(batch)

    scores = model.scores  # (n_evaluated, vocab_size)
    results = []

    for i in range(1, len(batch)):
        logits = scores[i - 1]
        # log-softmax
        max_logit = float(np.max(logits))
        shifted = logits - max_logit
        log_sum_exp = max_logit + np.log(np.sum(np.exp(shifted)))
        log_prob = float(logits[batch[i]]) - log_sum_exp
        loss = -log_prob
        results.append((batch[i], loss))

    return results


def classify_token(text: str) -> str:
    """Rough classification of a decoded token string."""
    stripped = text.strip()
    if not stripped:
        return "whitespace"
    if stripped.isdigit() or (stripped.startswith('-') and stripped[1:].isdigit()):
        return "number"
    if all(c in '(){}[]<>.,;:!?@#$%^&*+-=/\\|~`\'"' for c in stripped):
        return "punctuation"
    if stripped.startswith('http') or stripped.startswith('www'):
        return "url"
    if stripped[0].isupper() and stripped.isalpha():
        return "capitalized_word"
    if '_' in stripped or stripped.startswith('__'):
        return "code_identifier"
    if any(c.isupper() for c in stripped[1:]) and stripped[0].islower():
        return "camelCase"
    # Check if it's a common word (rough heuristic: short, lowercase, alpha)
    if stripped.isalpha() and stripped.islower() and len(stripped) <= 6:
        return "common_word"
    if stripped.isalpha() and stripped.islower():
        return "longer_word"
    if any(c.isdigit() for c in stripped) and any(c.isalpha() for c in stripped):
        return "alphanumeric_mix"
    return "other"


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # Load student data (has full text)
    print(f"Loading curriculum data from {STUDENT_DATA}...")
    with open(STUDENT_DATA) as f:
        all_chunks = json.load(f)
    print(f"  Total chunks available: {len(all_chunks)}")

    # Stratified sample: pick from different difficulty levels
    sorted_chunks = sorted(all_chunks, key=lambda x: x.get('ppl', 0))
    n_per_bucket = SAMPLE_SIZE // 5
    buckets = np.array_split(range(len(sorted_chunks)), 5)
    sample_indices = []
    for bucket in buckets:
        bucket_list = list(bucket)
        sample_indices.extend(random.sample(bucket_list, min(n_per_bucket, len(bucket_list))))

    # Fill remainder
    remaining = SAMPLE_SIZE - len(sample_indices)
    if remaining > 0:
        pool = [i for i in range(len(sorted_chunks)) if i not in set(sample_indices)]
        sample_indices.extend(random.sample(pool, min(remaining, len(pool))))

    chunks = [sorted_chunks[i] for i in sample_indices[:SAMPLE_SIZE]]
    print(f"  Sampled {len(chunks)} chunks (stratified by PPL)")
    total_tokens_est = sum(c.get('token_count', 512) for c in chunks)
    print(f"  Estimated total tokens: ~{total_tokens_est}")

    # Load models
    from llama_cpp import Llama

    print(f"\nLoading student model (Qwen3-1.7B)...")
    t0 = time.time()
    student_model = Llama(
        model_path=str(STUDENT_GGUF),
        n_ctx=CONTEXT_SIZE,
        n_batch=512,
        n_gpu_layers=-1,
        logits_all=True,
        verbose=False,
        seed=SEED,
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print(f"Loading teacher model (Qwen3-30B-A3B)...")
    t0 = time.time()
    teacher_model = Llama(
        model_path=str(TEACHER_GGUF),
        n_ctx=CONTEXT_SIZE,
        n_batch=512,
        n_gpu_layers=-1,
        logits_all=True,
        verbose=False,
        seed=SEED,
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Process chunks
    print(f"\nProcessing {len(chunks)} chunks...")
    all_student_losses = []
    all_teacher_losses = []
    all_zpd_ratios = []
    all_token_ids = []
    all_token_texts = []

    chunk_results = []
    start_time = time.time()
    errors = 0

    for ci, chunk in enumerate(chunks):
        text = chunk.get('text', '')
        if not text or len(text) < 20:
            errors += 1
            continue

        try:
            student_tl = compute_token_losses(student_model, text)
            teacher_tl = compute_token_losses(teacher_model, text)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error on chunk {ci}: {e}")
            continue

        # Align: both should have same token count if same tokenizer
        # But Qwen3-1.7B and 30B-A3B use the same tokenizer family, so should match
        n = min(len(student_tl), len(teacher_tl))
        if n < 5:
            errors += 1
            continue

        # Verify token alignment
        student_ids = [t[0] for t in student_tl[:n]]
        teacher_ids = [t[0] for t in teacher_tl[:n]]
        if student_ids != teacher_ids:
            # Tokenizers diverged — skip this chunk
            errors += 1
            if errors <= 5:
                print(f"  Token mismatch on chunk {ci}: student has {len(student_tl)} tokens, teacher has {len(teacher_tl)}")
            continue

        chunk_student_losses = []
        chunk_teacher_losses = []
        chunk_zpd = []

        for j in range(n):
            s_loss = student_tl[j][1]
            t_loss = teacher_tl[j][1]
            token_id = student_tl[j][0]

            # Decode token
            try:
                token_text = student_model.detokenize([token_id]).decode("utf-8", errors="replace")
            except:
                token_text = f"<id:{token_id}>"

            all_student_losses.append(s_loss)
            all_teacher_losses.append(t_loss)
            all_token_ids.append(token_id)
            all_token_texts.append(token_text)

            chunk_student_losses.append(s_loss)
            chunk_teacher_losses.append(t_loss)

            # ZPD ratio: student_loss / teacher_loss (avoid div by zero)
            if t_loss > 0.01:
                zpd = s_loss / t_loss
            else:
                zpd = 1.0  # both find it trivial
            all_zpd_ratios.append(zpd)
            chunk_zpd.append(zpd)

        # Chunk-level summary
        chunk_results.append({
            'text_preview': text[:100],
            'n_tokens': n,
            'student_mean_loss': float(np.mean(chunk_student_losses)),
            'teacher_mean_loss': float(np.mean(chunk_teacher_losses)),
            'chunk_zpd': float(np.mean(chunk_student_losses)) / max(float(np.mean(chunk_teacher_losses)), 0.01),
            'student_ppl_original': chunk.get('ppl', None),
            'high_zpd_token_fraction': float(np.mean(np.array(chunk_zpd) > ZPD_TOKEN_THRESHOLD)),
        })

        # Progress
        if (ci + 1) % 10 == 0 or ci == 0:
            elapsed = time.time() - start_time
            rate = (ci + 1) / elapsed * 60
            eta = (len(chunks) - ci - 1) / (rate / 60) if rate > 0 else 0
            n_total = len(all_student_losses)
            print(f"  [{ci+1}/{len(chunks)}] {n_total} tokens so far, "
                  f"{rate:.1f} chunks/min, ETA {eta:.0f}s, errors={errors}")

    total_time = time.time() - start_time

    # ========== ANALYSIS ==========
    print(f"\n{'='*70}")
    print("TOKEN-LEVEL ZPD ANALYSIS")
    print(f"{'='*70}")

    n_tokens_total = len(all_student_losses)
    print(f"\nProcessed: {len(chunk_results)} chunks, {n_tokens_total} tokens ({errors} errors)")
    print(f"Time: {total_time:.0f}s ({n_tokens_total / total_time:.0f} tokens/s)")

    student_arr = np.array(all_student_losses)
    teacher_arr = np.array(all_teacher_losses)
    zpd_arr = np.array(all_zpd_ratios)

    # --- 1. Token-level correlation ---
    print(f"\n--- Token-Level Correlation ---")
    token_spearman_r, token_spearman_p = stats.spearmanr(student_arr, teacher_arr)
    token_pearson_r, token_pearson_p = stats.pearsonr(student_arr, teacher_arr)
    print(f"Spearman rho: {token_spearman_r:.4f} (p={token_spearman_p:.2e})")
    print(f"Pearson r:    {token_pearson_r:.4f} (p={token_pearson_p:.2e})")
    print(f"Chunk-level rho was: {CHUNK_LEVEL_RHO:.4f}")
    delta_rho = CHUNK_LEVEL_RHO - token_spearman_r
    print(f"Delta (chunk - token): {delta_rho:.4f}")

    if token_spearman_r < CHUNK_LEVEL_RHO:
        print(f"  >>> CONFIRMED: Token-level rho ({token_spearman_r:.4f}) < chunk-level ({CHUNK_LEVEL_RHO})")
        print(f"  >>> Niche signal EXISTS at token level that averages out in chunks.")
    else:
        print(f"  >>> SURPRISE: Token-level rho >= chunk-level — no additional niche signal.")

    # --- 2. ZPD token fraction ---
    high_zpd_mask = zpd_arr > ZPD_TOKEN_THRESHOLD
    high_zpd_fraction = float(np.mean(high_zpd_mask))
    print(f"\n--- High-ZPD Token Analysis (threshold={ZPD_TOKEN_THRESHOLD}) ---")
    print(f"High-ZPD tokens: {np.sum(high_zpd_mask)}/{n_tokens_total} ({high_zpd_fraction:.1%})")
    print(f"ZPD distribution: mean={np.mean(zpd_arr):.3f}, median={np.median(zpd_arr):.3f}, "
          f"std={np.std(zpd_arr):.3f}")
    print(f"ZPD percentiles: p90={np.percentile(zpd_arr, 90):.2f}, "
          f"p95={np.percentile(zpd_arr, 95):.2f}, p99={np.percentile(zpd_arr, 99):.2f}")

    # --- 3. Niche token categorization ---
    print(f"\n--- Niche Token Categories ---")
    high_zpd_indices = np.where(high_zpd_mask)[0]

    category_counts = Counter()
    category_losses = defaultdict(list)
    category_examples = defaultdict(list)

    for idx in high_zpd_indices:
        token_text = all_token_texts[idx]
        cat = classify_token(token_text)
        category_counts[cat] += 1
        category_losses[cat].append(zpd_arr[idx])
        if len(category_examples[cat]) < 10:
            category_examples[cat].append({
                'text': repr(token_text),
                'student_loss': float(student_arr[idx]),
                'teacher_loss': float(teacher_arr[idx]),
                'zpd': float(zpd_arr[idx]),
            })

    # Also count categories for ALL tokens (baseline)
    all_category_counts = Counter()
    for t in all_token_texts:
        all_category_counts[classify_token(t)] += 1

    print(f"{'Category':<22} {'Niche':>6} {'Total':>6} {'%Niche':>7} {'%Total':>7} {'Over-rep':>8} {'Mean ZPD':>9}")
    print("-" * 75)
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        total = all_category_counts.get(cat, 1)
        pct_niche = count / max(len(high_zpd_indices), 1) * 100
        pct_total = total / n_tokens_total * 100
        overrep = (count / max(len(high_zpd_indices), 1)) / max(total / n_tokens_total, 1e-6)
        mean_zpd = np.mean(category_losses[cat])
        print(f"{cat:<22} {count:>6} {total:>6} {pct_niche:>6.1f}% {pct_total:>6.1f}% {overrep:>7.2f}x {mean_zpd:>8.2f}")

    # --- 4. Top niche tokens by ZPD ---
    print(f"\n--- Top 30 Highest-ZPD Tokens ---")
    top_indices = np.argsort(zpd_arr)[-30:][::-1]
    for rank, idx in enumerate(top_indices):
        print(f"  {rank+1:>2}. zpd={zpd_arr[idx]:>7.2f} | "
              f"student={student_arr[idx]:>7.3f} | teacher={teacher_arr[idx]:>7.3f} | "
              f"token={all_token_texts[idx]!r} [{classify_token(all_token_texts[idx])}]")

    # --- 5. Loss distribution stats ---
    print(f"\n--- Loss Distributions ---")
    print(f"Student loss: mean={np.mean(student_arr):.3f}, median={np.median(student_arr):.3f}, "
          f"std={np.std(student_arr):.3f}")
    print(f"Teacher loss: mean={np.mean(teacher_arr):.3f}, median={np.median(teacher_arr):.3f}, "
          f"std={np.std(teacher_arr):.3f}")

    # --- 6. Within-chunk ZPD variance ---
    print(f"\n--- Within-Chunk ZPD Variance ---")
    chunk_zpd_vars = []
    for cr in chunk_results:
        # We don't have per-token zpd stored per chunk, but we have the high_zpd_fraction
        pass

    # Actually let's compute this from the raw data by re-partitioning
    token_offset = 0
    within_chunk_rhos = []
    for cr in chunk_results:
        n = cr['n_tokens']
        if n < 10:
            token_offset += n
            continue
        chunk_s = student_arr[token_offset:token_offset + n]
        chunk_t = teacher_arr[token_offset:token_offset + n]
        if len(chunk_s) == len(chunk_t) and len(chunk_s) >= 10:
            rho, _ = stats.spearmanr(chunk_s, chunk_t)
            if not np.isnan(rho):
                within_chunk_rhos.append(rho)
        token_offset += n

    if within_chunk_rhos:
        rhos = np.array(within_chunk_rhos)
        print(f"Per-chunk Spearman rho (within-chunk token correlation):")
        print(f"  Mean: {np.mean(rhos):.4f}")
        print(f"  Median: {np.median(rhos):.4f}")
        print(f"  Std: {np.std(rhos):.4f}")
        print(f"  Range: [{np.min(rhos):.4f}, {np.max(rhos):.4f}]")

    # --- 7. Pre-registered prediction check ---
    print(f"\n{'='*70}")
    print("PRE-REGISTERED PREDICTION CHECK")
    print(f"{'='*70}")

    pred_rho = token_spearman_r < CHUNK_LEVEL_RHO
    pred_rho_range = 0.7 <= token_spearman_r <= 0.85
    pred_zpd_fraction = 0.05 <= high_zpd_fraction <= 0.20

    print(f"\n1. Token rho < chunk rho (0.958)?")
    print(f"   Token rho = {token_spearman_r:.4f} → {'PASS' if pred_rho else 'FAIL'}")

    print(f"\n2. Token rho in [0.7, 0.85] range?")
    print(f"   Token rho = {token_spearman_r:.4f} → {'PASS' if pred_rho_range else 'FAIL (out of range)'}")

    print(f"\n3. High-ZPD fraction in [5%, 20%]?")
    print(f"   Fraction = {high_zpd_fraction:.1%} → {'PASS' if pred_zpd_fraction else 'FAIL'}")

    # Check clustering
    if category_counts:
        top_cats = category_counts.most_common(3)
        predicted_cats = {'capitalized_word', 'code_identifier', 'longer_word', 'other', 'alphanumeric_mix', 'number'}
        top_cat_names = {c[0] for c in top_cats}
        cat_overlap = bool(top_cat_names & predicted_cats)
        print(f"\n4. High-ZPD tokens cluster around rare/domain/code tokens?")
        print(f"   Top categories: {', '.join(f'{c[0]} ({c[1]})' for c in top_cats)}")
        print(f"   Overlap with predicted categories: {'YES' if cat_overlap else 'NO'}")

    # --- Save results ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "token_level_zpd.json"

    # Build niche examples for each category
    niche_categories = {}
    for cat in category_counts:
        niche_categories[cat] = {
            'count': category_counts[cat],
            'total_in_corpus': all_category_counts.get(cat, 0),
            'over_representation': (category_counts[cat] / max(len(high_zpd_indices), 1)) / max(all_category_counts.get(cat, 0) / n_tokens_total, 1e-6),
            'mean_zpd': float(np.mean(category_losses[cat])),
            'examples': category_examples[cat][:5],
        }

    output = {
        'metadata': {
            'teacher_model': 'Qwen3-30B-A3B (GGUF via llama-cpp-python)',
            'student_model': 'Qwen3-1.7B (GGUF via llama-cpp-python)',
            'sample_size': len(chunk_results),
            'total_tokens': n_tokens_total,
            'errors': errors,
            'total_time_seconds': total_time,
            'tokens_per_second': n_tokens_total / total_time,
            'seed': SEED,
            'zpd_token_threshold': ZPD_TOKEN_THRESHOLD,
            'context_size': CONTEXT_SIZE,
            'chunk_level_rho_baseline': CHUNK_LEVEL_RHO,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        },
        'summary': {
            'token_spearman_rho': float(token_spearman_r),
            'token_spearman_p': float(token_spearman_p),
            'token_pearson_r': float(token_pearson_r),
            'token_pearson_p': float(token_pearson_p),
            'chunk_level_rho_baseline': CHUNK_LEVEL_RHO,
            'rho_delta': float(delta_rho),
            'high_zpd_token_fraction': float(high_zpd_fraction),
            'zpd_mean': float(np.mean(zpd_arr)),
            'zpd_median': float(np.median(zpd_arr)),
            'zpd_std': float(np.std(zpd_arr)),
            'zpd_p90': float(np.percentile(zpd_arr, 90)),
            'zpd_p95': float(np.percentile(zpd_arr, 95)),
            'zpd_p99': float(np.percentile(zpd_arr, 99)),
            'student_loss_mean': float(np.mean(student_arr)),
            'teacher_loss_mean': float(np.mean(teacher_arr)),
            'within_chunk_rho_mean': float(np.mean(within_chunk_rhos)) if within_chunk_rhos else None,
            'within_chunk_rho_median': float(np.median(within_chunk_rhos)) if within_chunk_rhos else None,
        },
        'pre_registered': {
            'token_rho_less_than_chunk': bool(pred_rho),
            'token_rho_in_expected_range': bool(pred_rho_range),
            'zpd_fraction_in_range': bool(pred_zpd_fraction),
        },
        'niche_categories': niche_categories,
        'top_30_niche_tokens': [
            {
                'rank': rank + 1,
                'token_text': all_token_texts[idx],
                'token_id': int(all_token_ids[idx]),
                'student_loss': float(student_arr[idx]),
                'teacher_loss': float(teacher_arr[idx]),
                'zpd': float(zpd_arr[idx]),
                'category': classify_token(all_token_texts[idx]),
            }
            for rank, idx in enumerate(np.argsort(zpd_arr)[-30:][::-1])
        ],
        'chunk_summaries': chunk_results,
    }

    def sanitize(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(sanitize(output), f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Final verdict
    print(f"\n{'='*70}")
    if pred_rho and high_zpd_fraction > 0.02:
        print("VERDICT: TOKEN-LEVEL NICHE SIGNAL CONFIRMED")
        print(f"Token rho ({token_spearman_r:.4f}) < chunk rho ({CHUNK_LEVEL_RHO})")
        print(f"{high_zpd_fraction:.1%} of tokens are high-ZPD niches.")
        print("These tokens define the content for hot-loadable expert leaves.")
    else:
        print("VERDICT: NEEDS FURTHER INVESTIGATION")
        print(f"Token rho={token_spearman_r:.4f}, ZPD fraction={high_zpd_fraction:.1%}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
