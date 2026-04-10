#!/usr/bin/env python3
"""EAGLE v3: feature fusion + multi-step training.

Key fix: at each draft step, concatenate head's output with the token embedding
of the predicted token. This grounds the head in real input space at every step,
preventing drift. This is EAGLE-2's core insight.

Architecture:
  Input at step t: concat(head_output_t-1, embed(predicted_token_t-1)) → project to hidden_dim
  → 1 transformer layer
  → predict token t

The token embedding is a FREE signal — it's just a lookup in the embedding table.
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


class EAGLEv3Head(nn.Module):
    """Draft head with token embedding fusion at each step."""

    def __init__(self, hidden_dim=4096, n_heads=8, ffn_dim=11008):
        super().__init__()
        # Fusion: concat(hidden_state, token_embed) → project to hidden_dim
        self.fuse = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)

        # Transformer layer
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True, dropout=0.0)
        self.norm2 = nn.RMSNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim, bias=False), nn.SiLU(),
            nn.Linear(ffn_dim, hidden_dim, bias=False))

    def forward(self, hidden_state, token_embed):
        """
        hidden_state: (batch, hidden_dim) — previous head output or real hidden state
        token_embed: (batch, hidden_dim) — embedding of the predicted/real token
        """
        # Fuse: ground hidden state with real token information
        x = self.fuse(torch.cat([hidden_state, token_embed], dim=-1))
        x = x.unsqueeze(1)  # (batch, 1, hidden_dim)

        # Transformer
        h = self.norm1(x)
        h = self.attn(h, h, h, need_weights=False)[0]
        x = x + h
        h = self.norm2(x)
        x = x + self.ffn(h)

        return x.squeeze(1)  # (batch, hidden_dim)


def collect_sequences(model, tok, texts, seq_len=64, max_seqs=3000):
    all_hs = []
    all_ids = []
    for i, text in enumerate(texts[:max_seqs]):
        ids = tok(text, return_tensors="pt", max_length=seq_len + 1,
                  truncation=True)["input_ids"].to("cuda:0")
        if ids.size(1) < 10: continue
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        all_hs.append(out.hidden_states[-1].squeeze(0).cpu())
        all_ids.append(ids.squeeze(0).cpu())
        if (i + 1) % 500 == 0:
            print("  Collected %d/%d" % (i + 1, len(texts[:max_seqs])))
    return all_hs, all_ids


def main():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map={"": 0})
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model.eval()

    embed = model.model.embed_tokens  # token → hidden_dim lookup
    final_norm = model.model.norm
    lm_head = model.lm_head
    embed.requires_grad_(False)
    final_norm.requires_grad_(False)
    lm_head.requires_grad_(False)

    # Collect data
    print("Collecting sequences...")
    ruby = [json.loads(l)["text"] for l in open("/root/t6b-mogae/training_data/ruby_domain.jsonl")]
    generic = [json.loads(l)["text"] for l in open("/root/t6b-mogae/training_data/generic.jsonl")]
    texts = ruby[:2000] + generic[:1000]
    all_hs, all_ids = collect_sequences(model, tok, texts)
    print("Collected %d sequences" % len(all_hs))

    # Free layers
    del model.model.layers
    import gc; gc.collect(); torch.cuda.empty_cache()

    # Build head (initialize from single-step head weights where possible)
    head = EAGLEv3Head(hidden_dim=4096, n_heads=8, ffn_dim=11008)
    # Load single-step weights into matching layers
    old_state = torch.load("/root/t6b-mogae/eagle_draft_head.pt", map_location="cpu", weights_only=True)
    new_state = head.state_dict()
    loaded = 0
    for key in old_state:
        # Map old keys: norm1→norm1, attn→attn, norm2→norm2, ffn→ffn
        if key in new_state and old_state[key].shape == new_state[key].shape:
            new_state[key] = old_state[key]
            loaded += 1
    head.load_state_dict(new_state)
    head = head.to(device="cuda:0", dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print("Head: %dM params (%d layers loaded from single-step)" % (n_params // 1_000_000, loaded))

    # Training
    k_steps = 6
    n_steps = 5000
    eval_every = 500
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    n_eval = min(200, len(all_hs) // 10)
    train_hs, train_ids = all_hs[:-n_eval], all_ids[:-n_eval]
    eval_hs, eval_ids = all_hs[-n_eval:], all_ids[-n_eval:]

    print("\nTraining (k=%d, %d steps, with token embedding fusion)..." % (k_steps, n_steps))
    t0 = time.time()
    best_acc = 0
    losses_log = []

    for step in range(n_steps):
        head.train()
        total_loss = torch.tensor(0.0, device="cuda:0")
        n_tokens = 0

        # Mini-batch: 8 sequences per step
        seq_indices = torch.randint(0, len(train_hs), (8,))
        for si in seq_indices:
            hs_seq = train_hs[si].to("cuda:0")
            id_seq = train_ids[si].to("cuda:0")
            if hs_seq.size(0) < k_steps + 2: continue

            max_start = hs_seq.size(0) - k_steps - 1
            if max_start < 1: continue
            start = torch.randint(0, max_start, (1,)).item()

            # Start from real hidden state
            h = hs_seq[start:start+1]  # (1, hidden_dim)

            for j in range(k_steps):
                target_pos = start + j + 1
                if target_pos >= id_seq.size(0): break

                # Token embedding of the token AT the current position
                # (this is what the model "saw" to produce this hidden state)
                tok_id = id_seq[start + j]
                tok_emb = embed(tok_id.unsqueeze(0))  # (1, hidden_dim)

                # Head forward with fusion
                h = head(h, tok_emb)

                # Predict
                logits = lm_head(final_norm(h))
                target = id_seq[target_pos:target_pos+1]
                total_loss = total_loss + F.cross_entropy(logits, target)
                n_tokens += 1

        if n_tokens == 0: continue
        avg_loss = total_loss / n_tokens
        optimizer.zero_grad()
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses_log.append(avg_loss.item())

        if (step + 1) % eval_every == 0:
            head.eval()
            correct_per = [0] * k_steps
            total_per = [0] * k_steps

            with torch.no_grad():
                for ei in range(min(100, len(eval_hs))):
                    hs_seq = eval_hs[ei].to("cuda:0")
                    id_seq = eval_ids[ei].to("cuda:0")
                    if hs_seq.size(0) < k_steps + 2: continue

                    for start in range(0, hs_seq.size(0) - k_steps - 1, 8):
                        h = hs_seq[start:start+1]
                        for j in range(k_steps):
                            target_pos = start + j + 1
                            if target_pos >= id_seq.size(0): break
                            tok_id = id_seq[start + j]
                            tok_emb = embed(tok_id.unsqueeze(0))
                            h = head(h, tok_emb)
                            logits = lm_head(final_norm(h))
                            pred = logits.argmax(dim=-1).item()
                            if pred == id_seq[target_pos].item():
                                correct_per[j] += 1
                            total_per[j] += 1

            accs = [correct_per[j] / max(total_per[j], 1) for j in range(k_steps)]
            overall = sum(correct_per) / max(sum(total_per), 1)
            avg_l = sum(losses_log[-eval_every:]) / len(losses_log[-eval_every:])

            if overall > best_acc:
                best_acc = overall
                torch.save(head.state_dict(), "/root/t6b-mogae/eagle_v3_head.pt")
                tag = " *BEST*"
            else:
                tag = ""

            acc_str = " ".join(["%.0f%%" % (a*100) for a in accs])
            print("  Step %d: loss=%.3f overall=%.1f%% steps=[%s] (%.0fs)%s" % (
                step+1, avg_l, overall*100, acc_str, time.time()-t0, tag))

    print("\nDone in %.0fs. Best: %.1f%%" % (time.time()-t0, best_acc*100))
    torch.save(head.state_dict(), "/root/t6b-mogae/eagle_v3_head_final.pt")


if __name__ == "__main__":
    main()
