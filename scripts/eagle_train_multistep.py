#!/usr/bin/env python3
"""EAGLE multi-step training: teach the draft head to draft autoregressively.

Key difference from single-step: backprop through the head's own k-step chain.
The head learns to correct its own drift by seeing what happens when it feeds
its output back as input.

Training procedure:
1. Get real hidden states from target model at positions t, t+1, ..., t+k
2. Head predicts from real h_t → draft_h_t+1 → draft_h_t+2 → ... (autoregressive)
3. Loss: cross-entropy at EACH position against target model's token
4. Backprop through the entire chain → head learns to compensate for drift

This is the core of EAGLE-2/3's training strategy.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
sys.path.insert(0, "/root/t6b-mogae")

import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


class EAGLEDraftHead(nn.Module):
    def __init__(self, hidden_dim=4096, n_heads=8, ffn_dim=11008):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True, dropout=0.0)
        self.norm2 = nn.RMSNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim, bias=False), nn.SiLU(),
            nn.Linear(ffn_dim, hidden_dim, bias=False))
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.proj(x)
        h = self.norm1(x)
        h = self.attn(h, h, h, need_weights=False)[0]
        x = x + h
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x.squeeze(1)


def collect_sequences(model, tok, texts, seq_len=64, max_seqs=3000):
    """Collect hidden state sequences from the target model.

    Returns (hidden_states, token_ids) where:
    - hidden_states: (n_seqs, seq_len, hidden_dim) — last layer hidden states
    - token_ids: (n_seqs, seq_len) — token ids at each position
    """
    device = "cuda:0"
    all_hs = []
    all_ids = []

    for i, text in enumerate(texts[:max_seqs]):
        ids = tok(text, return_tensors="pt", max_length=seq_len + 1,
                  truncation=True)["input_ids"].to(device)
        if ids.size(1) < 10:
            continue

        with torch.no_grad():
            out = model(ids, output_hidden_states=True)

        hs = out.hidden_states[-1].squeeze(0).cpu()  # (seq_len, hidden_dim)
        tids = ids.squeeze(0).cpu()  # (seq_len,)

        all_hs.append(hs)
        all_ids.append(tids)

        if (i + 1) % 500 == 0:
            print("  Collected %d/%d sequences" % (i + 1, len(texts[:max_seqs])))

    return all_hs, all_ids


def main():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map={"": 0})
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model.eval()

    final_norm = model.model.norm
    lm_head = model.lm_head
    final_norm.requires_grad_(False)
    lm_head.requires_grad_(False)

    # Load pre-trained single-step head as starting point
    head = EAGLEDraftHead(hidden_dim=4096, n_heads=8, ffn_dim=11008)
    head.load_state_dict(torch.load("/root/t6b-mogae/eagle_draft_head.pt",
                                     map_location="cpu", weights_only=True))
    head = head.to(device="cuda:0", dtype=torch.bfloat16)
    print("Loaded single-step head (77% accuracy) as initialization")

    # Collect sequence data
    print("\nCollecting hidden state sequences...")
    ruby_texts = []
    with open("/root/t6b-mogae/training_data/ruby_domain.jsonl") as f:
        for line in f:
            ruby_texts.append(json.loads(line)["text"])
    generic_texts = []
    with open("/root/t6b-mogae/training_data/generic.jsonl") as f:
        for line in f:
            generic_texts.append(json.loads(line)["text"])

    texts = ruby_texts[:2000] + generic_texts[:1000]
    all_hs, all_ids = collect_sequences(model, tok, texts, seq_len=64, max_seqs=3000)
    print("Collected %d sequences" % len(all_hs))

    # Free model layers (keep only norm + lm_head)
    del model.model.layers
    import gc; gc.collect(); torch.cuda.empty_cache()

    # Multi-step training
    k_steps = 6  # draft chain length during training
    batch_size = 64
    n_steps = 5000
    eval_every = 500

    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    # Split train/eval
    n_eval = min(200, len(all_hs) // 10)
    train_hs, train_ids = all_hs[:-n_eval], all_ids[:-n_eval]
    eval_hs, eval_ids = all_hs[-n_eval:], all_ids[-n_eval:]

    print("\nMulti-step training (k=%d, %d steps)..." % (k_steps, n_steps))
    t0 = time.time()
    best_acc = 0
    losses_log = []

    for step in range(n_steps):
        head.train()

        # Sample batch of sequences and random starting positions
        total_loss = 0.0
        n_tokens = 0

        for _ in range(batch_size // 8):  # mini-batches of 8 sequences
            # Pick random sequences
            seq_indices = torch.randint(0, len(train_hs), (8,))

            for si in seq_indices:
                hs_seq = train_hs[si].to("cuda:0")  # (seq_len, hidden_dim)
                id_seq = train_ids[si].to("cuda:0")  # (seq_len,)

                if hs_seq.size(0) < k_steps + 2:
                    continue

                # Random start position (leave room for k steps)
                max_start = hs_seq.size(0) - k_steps - 1
                if max_start < 1:
                    continue
                start = torch.randint(0, max_start, (1,)).item()

                # Autoregressive chain through the head
                # Start from REAL hidden state at position `start`
                h = hs_seq[start:start+1]  # (1, hidden_dim)
                chain_loss = torch.tensor(0.0, device="cuda:0")

                for j in range(k_steps):
                    target_pos = start + j + 1
                    if target_pos >= id_seq.size(0):
                        break

                    # Head forward
                    h = head(h)  # (1, hidden_dim)

                    # Predict token
                    logits = lm_head(final_norm(h))  # (1, vocab_size)
                    target = id_seq[target_pos:target_pos+1]

                    # Loss with position-dependent weight:
                    # later positions harder → equal weight forces head to learn drift correction
                    chain_loss = chain_loss + F.cross_entropy(logits, target)
                    n_tokens += 1

                total_loss = total_loss + chain_loss

        if n_tokens == 0:
            continue

        avg_loss = total_loss / n_tokens
        optimizer.zero_grad()
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses_log.append(avg_loss.item())

        # Eval
        if (step + 1) % eval_every == 0:
            head.eval()
            correct_per_step = [0] * k_steps
            total_per_step = [0] * k_steps

            with torch.no_grad():
                for ei in range(min(100, len(eval_hs))):
                    hs_seq = eval_hs[ei].to("cuda:0")
                    id_seq = eval_ids[ei].to("cuda:0")
                    if hs_seq.size(0) < k_steps + 2:
                        continue

                    # Test from multiple starting points
                    for start in range(0, hs_seq.size(0) - k_steps - 1, 8):
                        h = hs_seq[start:start+1]
                        for j in range(k_steps):
                            target_pos = start + j + 1
                            if target_pos >= id_seq.size(0):
                                break
                            h = head(h)
                            logits = lm_head(final_norm(h))
                            pred = logits.argmax(dim=-1).item()
                            target = id_seq[target_pos].item()
                            if pred == target:
                                correct_per_step[j] += 1
                            total_per_step[j] += 1

            accs = []
            for j in range(k_steps):
                if total_per_step[j] > 0:
                    acc = correct_per_step[j] / total_per_step[j]
                    accs.append(acc)

            avg_loss_recent = sum(losses_log[-eval_every:]) / len(losses_log[-eval_every:])
            overall_acc = sum(correct_per_step) / max(sum(total_per_step), 1)

            if overall_acc > best_acc:
                best_acc = overall_acc
                torch.save(head.state_dict(), "/root/t6b-mogae/eagle_draft_head_multistep.pt")
                tag = " *BEST*"
            else:
                tag = ""

            acc_str = " ".join(["%.0f%%" % (a*100) for a in accs])
            print("  Step %d: loss=%.3f overall=%.1f%% per-step=[%s] (%.0fs)%s" % (
                step + 1, avg_loss_recent, overall_acc * 100, acc_str, time.time() - t0, tag))

    print("\nDone in %.0fs. Best overall: %.1f%%" % (time.time() - t0, best_acc * 100))
    torch.save(head.state_dict(), "/root/t6b-mogae/eagle_draft_head_multistep_final.pt")


if __name__ == "__main__":
    main()
