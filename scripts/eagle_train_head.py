#!/usr/bin/env python3
"""Train EAGLE-3 draft head: 1 transformer layer that predicts next token from hidden states.

Architecture:
  Input: last hidden state from Qwen3-8B (4096 dim)
  → 1 transformer decoder layer (4096 dim, 8 heads, 11008 FFN)
  → Reuse Qwen3-8B's LM head (frozen) for logits
  → Cross-entropy loss against actual next token

~33M trainable params. Training: ~10 min on 500K tokens.
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
    """Single transformer layer that predicts next token from hidden states."""

    def __init__(self, hidden_dim=4096, n_heads=8, ffn_dim=11008):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, n_heads, batch_first=True, dropout=0.0)
        self.norm2 = nn.RMSNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim, bias=False),
            nn.SiLU(),
            nn.Linear(ffn_dim, hidden_dim, bias=False),
        )
        # Feature projection: transforms "where the model is" into
        # "what the model will say next"
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        """x: (batch, hidden_dim) or (batch, seq, hidden_dim)"""
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, hidden_dim)

        # Project: shift from "current state" to "next prediction" space
        x = self.proj(x)

        # Self-attention (single position → no mask needed)
        h = self.norm1(x)
        h = self.attn(h, h, h, need_weights=False)[0]
        x = x + h

        # FFN
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h

        return x.squeeze(1)  # (batch, hidden_dim)


def main():
    print("Loading training data...")
    data = torch.load("/root/t6b-mogae/eagle_train_data.pt", weights_only=True)
    H = data["hidden_states"]  # (N, 4096) bf16
    L = data["labels"]  # (N,) long
    print(f"Data: {H.shape[0]} tokens, {H.shape[1]} dim")

    print("Loading LM head from Qwen3-8B...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map={"": 0})
    lm_head = model.lm_head  # (vocab_size, hidden_dim)
    lm_head.requires_grad_(False)
    # Also grab the final norm
    final_norm = model.model.norm
    final_norm.requires_grad_(False)

    # Free the rest of the model
    del model.model.layers
    import gc; gc.collect(); torch.cuda.empty_cache()

    # Build draft head
    head = EAGLEDraftHead(hidden_dim=4096, n_heads=8, ffn_dim=11008)
    head = head.to(device="cuda:0", dtype=torch.bfloat16)

    n_params = sum(p.numel() for p in head.parameters())
    print(f"Draft head: {n_params/1e6:.1f}M params")

    # Training setup
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)
    batch_size = 512
    n_steps = 5000
    eval_every = 500

    # Split: 90% train, 10% eval
    n_eval = min(50000, H.shape[0] // 10)
    H_eval, L_eval = H[-n_eval:].to("cuda:0"), L[-n_eval:].to("cuda:0")
    H_train, L_train = H[:-n_eval], L[:-n_eval]

    print(f"Train: {H_train.shape[0]}, Eval: {H_eval.shape[0]}")
    print(f"Training for {n_steps} steps...")

    t0 = time.time()
    best_acc = 0
    losses = []

    for step in range(n_steps):
        idx = torch.randint(0, H_train.shape[0], (batch_size,))
        h_batch = H_train[idx].to("cuda:0")
        l_batch = L_train[idx].to("cuda:0")

        # Forward: draft head → norm → lm_head
        draft_hs = head(h_batch)
        draft_hs = final_norm(draft_hs)
        logits = lm_head(draft_hs)  # (batch, vocab_size)

        loss = F.cross_entropy(logits, l_batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if (step + 1) % eval_every == 0:
            # Eval
            head.eval()
            with torch.no_grad():
                # Process eval in chunks
                correct = 0
                total = 0
                for i in range(0, H_eval.shape[0], 2048):
                    h_e = H_eval[i:i+2048]
                    l_e = L_eval[i:i+2048]
                    draft_hs = head(h_e)
                    draft_hs = final_norm(draft_hs)
                    logits = lm_head(draft_hs)
                    preds = logits.argmax(dim=-1)
                    correct += (preds == l_e).sum().item()
                    total += l_e.shape[0]

                acc = correct / total
                avg_loss = sum(losses[-eval_every:]) / eval_every
                elapsed = time.time() - t0

                if acc > best_acc:
                    best_acc = acc
                    torch.save(head.state_dict(), "/root/t6b-mogae/eagle_draft_head.pt")
                    tag = " *BEST*"
                else:
                    tag = ""

                print(f"  Step {step+1}: loss={avg_loss:.4f} acc={acc:.1%} "
                      f"best={best_acc:.1%} ({elapsed:.0f}s){tag}")
            head.train()

    # Final save
    torch.save(head.state_dict(), "/root/t6b-mogae/eagle_draft_head_final.pt")

    print(f"\nTraining done in {time.time()-t0:.0f}s")
    print(f"Best accuracy: {best_acc:.1%}")
    print(f"Saved to eagle_draft_head.pt (best) and eagle_draft_head_final.pt")


if __name__ == "__main__":
    main()
