#!/usr/bin/env python3
"""
Teacher-driven curriculum builder: score 500 C4 chunks at TOKEN level,
bucket them into niche-heavy / mixed / easy for MoE expert training.

Uses the same per-token ZPD approach as token_level_zpd.py but scaled 10x.
Output: results/teacher_curriculum.json with 3 buckets for expert routing.

Buckets:
  - niche_heavy (>15% high-ZPD tokens): teacher knows much more than student
  - mixed (5-15%): moderate niche content
  - easy (<5%): both models handle well

Training plan:
  - Expert 0: niche-heavy chunks
  - Expert 1: easy chunks
  - Expert 2 (shared): all chunks
"""

import json
import math
import time
import random
import numpy as np
from pathlib import Path

# --- Configuration ---
TEACHER_GGUF = Path.home() / ".ollama/models/blobs/sha256-58574f2e94b99fb9e4391408b57e5aeaaaec10f6384e9a699fc2cb43a5c8eabf"
STUDENT_GGUF = Path.home() / ".ollama/models/blobs/sha256-3d0b790534fe4b79525fc3692950408dca41171676ed7e21db57af5c65ef6ab6"
STUDENT_DATA = Path("/tmp/curriculum_scored.json")
RESULTS_DIR = Path.home() / "github.com/erikdebruijn/tree-of-knowledge-llm/results"

SAMPLE_SIZE = 500
CONTEXT_SIZE = 2048
SEED = 42
ZPD_TOKEN_THRESHOLD = 3.0

# Bucket thresholds (fraction of high-ZPD tokens in a chunk)
NICHE_HEAVY_THRESHOLD = 0.15  # >15%
MIXED_THRESHOLD = 0.05        # 5-15%
# easy = <5%


def compute_token_losses(model, text: str) -> list[tuple[int, float]]:
    """Compute per-token cross-entropy loss.
    Returns list of (token_id, loss) for each predicted token (skips BOS).
    """
    tokens = model.tokenize(text.encode("utf-8"), add_bos=True)
    n_tokens = len(tokens)
    if n_tokens < 2:
        return []

    batch_size = min(n_tokens, CONTEXT_SIZE)
    batch = tokens[:batch_size]

    model.reset()
    model.eval(batch)

    scores = model.scores
    results = []

    for i in range(1, len(batch)):
        logits = scores[i - 1]
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

    # --- Load data ---
    print(f"Loading curriculum data from {STUDENT_DATA}...")
    with open(STUDENT_DATA) as f:
        all_chunks = json.load(f)
    print(f"  Total chunks available: {len(all_chunks)}")

    # Stratified sample: 500 chunks across PPL range
    sorted_chunks = sorted(all_chunks, key=lambda x: x.get('ppl', 0))
    n_buckets = 10
    n_per_bucket = SAMPLE_SIZE // n_buckets
    buckets = np.array_split(range(len(sorted_chunks)), n_buckets)
    sample_indices = []
    for bucket in buckets:
        bucket_list = list(bucket)
        sample_indices.extend(random.sample(bucket_list, min(n_per_bucket, len(bucket_list))))

    remaining = SAMPLE_SIZE - len(sample_indices)
    if remaining > 0:
        pool = [i for i in range(len(sorted_chunks)) if i not in set(sample_indices)]
        sample_indices.extend(random.sample(pool, min(remaining, len(pool))))

    chunks = [sorted_chunks[i] for i in sample_indices[:SAMPLE_SIZE]]
    total_tokens_est = sum(c.get('token_count', 512) for c in chunks)
    print(f"  Sampled {len(chunks)} chunks (stratified by PPL across {n_buckets} buckets)")
    print(f"  Estimated total tokens: ~{total_tokens_est}")
    print(f"  Estimated time at 444 tok/s: ~{total_tokens_est / 444:.0f}s ({total_tokens_est / 444 / 60:.1f} min)")

    # --- Load models ---
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

    # --- Process chunks ---
    print(f"\nProcessing {len(chunks)} chunks...")
    chunk_results = []
    errors = 0
    total_tokens_processed = 0
    start_time = time.time()

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

        n = min(len(student_tl), len(teacher_tl))
        if n < 5:
            errors += 1
            continue

        # Verify token alignment
        student_ids = [t[0] for t in student_tl[:n]]
        teacher_ids = [t[0] for t in teacher_tl[:n]]
        if student_ids != teacher_ids:
            errors += 1
            if errors <= 5:
                print(f"  Token mismatch on chunk {ci}")
            continue

        # Compute per-token ZPD
        zpd_values = []
        high_zpd_tokens = []
        student_losses = []
        teacher_losses = []

        for j in range(n):
            s_loss = student_tl[j][1]
            t_loss = teacher_tl[j][1]
            token_id = student_tl[j][0]

            student_losses.append(s_loss)
            teacher_losses.append(t_loss)

            if t_loss > 0.01:
                zpd = s_loss / t_loss
            else:
                zpd = 1.0

            zpd_values.append(zpd)

            if zpd > ZPD_TOKEN_THRESHOLD:
                try:
                    token_text = student_model.detokenize([token_id]).decode("utf-8", errors="replace")
                except Exception:
                    token_text = f"<id:{token_id}>"
                high_zpd_tokens.append({
                    'token_id': int(token_id),
                    'token_text': token_text,
                    'student_loss': float(s_loss),
                    'teacher_loss': float(t_loss),
                    'zpd': float(zpd),
                    'category': classify_token(token_text),
                })

        high_zpd_fraction = len(high_zpd_tokens) / n if n > 0 else 0.0

        # Determine bucket
        if high_zpd_fraction > NICHE_HEAVY_THRESHOLD:
            bucket = "niche_heavy"
        elif high_zpd_fraction > MIXED_THRESHOLD:
            bucket = "mixed"
        else:
            bucket = "easy"

        chunk_result = {
            'chunk_index': ci,
            'text': text,
            'n_tokens': n,
            'student_mean_loss': float(np.mean(student_losses)),
            'teacher_mean_loss': float(np.mean(teacher_losses)),
            'zpd_mean': float(np.mean(zpd_values)),
            'zpd_median': float(np.median(zpd_values)),
            'high_zpd_fraction': float(high_zpd_fraction),
            'n_high_zpd_tokens': len(high_zpd_tokens),
            'bucket': bucket,
            'original_ppl': chunk.get('ppl', None),
            'original_difficulty_percentile': chunk.get('difficulty_percentile', None),
            'high_zpd_token_categories': {},
        }

        # Summarize niche token categories for this chunk
        from collections import Counter
        cat_counts = Counter(t['category'] for t in high_zpd_tokens)
        chunk_result['high_zpd_token_categories'] = dict(cat_counts)

        chunk_results.append(chunk_result)
        total_tokens_processed += n

        # Progress every 25 chunks
        if (ci + 1) % 25 == 0 or ci == 0:
            elapsed = time.time() - start_time
            rate = total_tokens_processed / elapsed if elapsed > 0 else 0
            eta = (total_tokens_est - total_tokens_processed) / rate if rate > 0 else 0
            buckets_so_far = Counter(cr['bucket'] for cr in chunk_results)
            print(f"  [{ci+1}/{len(chunks)}] {total_tokens_processed} tokens, "
                  f"{rate:.0f} tok/s, ETA {eta:.0f}s | "
                  f"niche={buckets_so_far.get('niche_heavy', 0)} "
                  f"mixed={buckets_so_far.get('mixed', 0)} "
                  f"easy={buckets_so_far.get('easy', 0)} | "
                  f"errors={errors}")

    total_time = time.time() - start_time

    # --- Bucket the results ---
    from collections import Counter
    niche_heavy = [cr for cr in chunk_results if cr['bucket'] == 'niche_heavy']
    mixed = [cr for cr in chunk_results if cr['bucket'] == 'mixed']
    easy = [cr for cr in chunk_results if cr['bucket'] == 'easy']

    print(f"\n{'='*70}")
    print("TEACHER CURRICULUM SUMMARY")
    print(f"{'='*70}")
    print(f"Total chunks processed: {len(chunk_results)} ({errors} errors)")
    print(f"Total tokens: {total_tokens_processed}")
    print(f"Time: {total_time:.0f}s ({total_tokens_processed / max(total_time, 1):.0f} tok/s)")
    print(f"\nBucket distribution:")
    print(f"  niche_heavy (>{NICHE_HEAVY_THRESHOLD:.0%} high-ZPD): {len(niche_heavy)} chunks")
    print(f"  mixed ({MIXED_THRESHOLD:.0%}-{NICHE_HEAVY_THRESHOLD:.0%} high-ZPD):     {len(mixed)} chunks")
    print(f"  easy (<{MIXED_THRESHOLD:.0%} high-ZPD):                   {len(easy)} chunks")

    # Per-bucket stats
    for name, bucket_chunks in [("niche_heavy", niche_heavy), ("mixed", mixed), ("easy", easy)]:
        if not bucket_chunks:
            continue
        fracs = [c['high_zpd_fraction'] for c in bucket_chunks]
        ppls = [c['original_ppl'] for c in bucket_chunks if c['original_ppl'] is not None]
        tokens = sum(c['n_tokens'] for c in bucket_chunks)
        print(f"\n  {name}:")
        print(f"    Chunks: {len(bucket_chunks)}, Tokens: {tokens}")
        print(f"    High-ZPD fraction: mean={np.mean(fracs):.3f}, range=[{np.min(fracs):.3f}, {np.max(fracs):.3f}]")
        if ppls:
            print(f"    Original PPL: mean={np.mean(ppls):.2f}, median={np.median(ppls):.2f}")

        # Aggregate niche categories
        all_cats = Counter()
        for c in bucket_chunks:
            for cat, cnt in c.get('high_zpd_token_categories', {}).items():
                all_cats[cat] += cnt
        if all_cats:
            top5 = all_cats.most_common(5)
            print(f"    Top niche categories: {', '.join(f'{c}({n})' for c, n in top5)}")

    # --- Save curriculum ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "teacher_curriculum.json"

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

    curriculum = {
        'metadata': {
            'teacher_model': 'Qwen3-30B-A3B (GGUF via llama-cpp-python)',
            'student_model': 'Qwen3-1.7B (GGUF via llama-cpp-python)',
            'sample_size': len(chunk_results),
            'total_tokens': total_tokens_processed,
            'errors': errors,
            'total_time_seconds': total_time,
            'tokens_per_second': total_tokens_processed / max(total_time, 1),
            'seed': SEED,
            'zpd_token_threshold': ZPD_TOKEN_THRESHOLD,
            'niche_heavy_threshold': NICHE_HEAVY_THRESHOLD,
            'mixed_threshold': MIXED_THRESHOLD,
            'context_size': CONTEXT_SIZE,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        },
        'summary': {
            'n_niche_heavy': len(niche_heavy),
            'n_mixed': len(mixed),
            'n_easy': len(easy),
            'niche_heavy_tokens': sum(c['n_tokens'] for c in niche_heavy),
            'mixed_tokens': sum(c['n_tokens'] for c in mixed),
            'easy_tokens': sum(c['n_tokens'] for c in easy),
        },
        'training_plan': {
            'expert_0': 'niche_heavy chunks — teacher-student gap is large',
            'expert_1': 'easy chunks — both models agree',
            'expert_2_shared': 'all chunks — general capacity',
            'routing': 'gumbel-softmax, damage surrogate loss',
        },
        'buckets': {
            'niche_heavy': niche_heavy,
            'mixed': mixed,
            'easy': easy,
        },
    }

    with open(output_path, 'w') as f:
        json.dump(sanitize(curriculum), f, indent=2)

    print(f"\nCurriculum saved to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"\nReady for Step 2: transfer to GPU server and train experts.")


if __name__ == '__main__':
    main()
