#!/usr/bin/env python3
"""EAGLE-3 proper training: exact architecture from SafeAILab/EAGLE.

Key insights from their code that our v1-v3 missed:
1. Input = concat(token_embedding, hidden_state) → 2*hidden_dim input to attention
2. KV cache across autoregressive steps (attention builds context)
3. Loss = KL divergence against target model's DISTRIBUTION (soft targets)
4. Multi-layer feature fusion: concat(hs[0], hs[1], hs[2]) → project to hidden_dim
5. 7 autoregressive steps per training example

This script: single-GPU, no DeepSpeed, Qwen3-8B specific.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
sys.path.insert(0, "/root/t6b-mogae")

import json, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        orig_dtype = x.dtype
        norm = x.float().pow(2).mean(-1, keepdim=True)
        return ((x * torch.rsqrt(norm + self.eps)) * self.weight).to(orig_dtype)


class EAGLELayer(nn.Module):
    """Single EAGLE decoder layer: takes (token_emb, hidden_state) concatenated."""

    def __init__(self, hidden_dim=4096, n_heads=32, n_kv_heads=8, ffn_dim=14336, eps=1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = hidden_dim // n_heads

        # Input is 2*hidden_dim (concat of emb + hidden), needs projection for Q
        # But K,V come from the concat too
        self.q_proj = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(2 * hidden_dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(2 * hidden_dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.hidden_norm = RMSNorm(hidden_dim, eps)
        self.input_norm = RMSNorm(hidden_dim, eps)
        self.post_attn_norm = RMSNorm(hidden_dim, eps)

        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, token_emb, hidden_state, kv_cache=None):
        """
        token_emb: (B, 1, hidden_dim) — embedding of previous token
        hidden_state: (B, 1, hidden_dim) — previous step's output
        kv_cache: (k_cache, v_cache) or None

        Returns: (output, new_kv_cache)
        """
        residual = hidden_state

        # Normalize and concatenate
        h_norm = self.hidden_norm(hidden_state)
        e_norm = self.input_norm(token_emb)
        combined = torch.cat([e_norm, h_norm], dim=-1)  # (B, 1, 2*hidden_dim)

        B, L, _ = combined.shape

        # QKV from combined input
        q = self.q_proj(combined).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(combined).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(combined).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # KV cache: append new K,V to cache
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        new_kv_cache = (k, v)

        # GQA repeat
        n_rep = self.n_heads // self.n_kv_heads
        if n_rep > 1:
            k = k[:, :, None, :, :].expand(B, self.n_kv_heads, n_rep, -1, self.head_dim)
            k = k.reshape(B, self.n_heads, -1, self.head_dim)
            v = v[:, :, None, :, :].expand(B, self.n_kv_heads, n_rep, -1, self.head_dim)
            v = v.reshape(B, self.n_heads, -1, self.head_dim)

        # Attention
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, -1)
        attn_out = self.o_proj(attn_out)

        hidden_state = residual + attn_out

        # MLP
        residual = hidden_state
        h = self.post_attn_norm(hidden_state)
        hidden_state = residual + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))

        return hidden_state, new_kv_cache


class EAGLE3Head(nn.Module):
    """Complete EAGLE-3 draft head for Qwen3-8B."""

    def __init__(self, hidden_dim=4096, n_heads=32, n_kv_heads=8, ffn_dim=14336):
        super().__init__()
        # Feature fusion: 3 hidden states from target model
        self.fc = nn.Linear(hidden_dim * 3, hidden_dim, bias=False)
        # Single decoder layer
        self.layer = EAGLELayer(hidden_dim, n_heads, n_kv_heads, ffn_dim)
        # Output norm (reuse target model's norm at inference)
        self.norm = RMSNorm(hidden_dim)

    def forward(self, token_emb, fused_hidden, kv_cache=None):
        """One draft step.

        Args:
            token_emb: (B, 1, hidden_dim) — embedding of the token
            fused_hidden: (B, 1, hidden_dim) — fused hidden state (or previous output)
            kv_cache: tuple of (k, v) caches or None

        Returns: (hidden_out, logits, new_kv_cache)
        """
        hidden_out, new_kv = self.layer(token_emb, fused_hidden, kv_cache)
        normed = self.norm(hidden_out)
        return hidden_out, normed, new_kv


def main():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map={"": 0})
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model.eval()

    config = model.config
    embed = model.model.embed_tokens
    lm_head = model.lm_head
    embed.requires_grad_(False)
    lm_head.requires_grad_(False)

    # Collect training data: full sequences with hidden states from 3 layers
    print("Collecting training data...")
    ruby = [json.loads(l)["text"] for l in open("/root/t6b-mogae/training_data/ruby_domain.jsonl")]
    generic = [json.loads(l)["text"] for l in open("/root/t6b-mogae/training_data/generic.jsonl")]
    texts = ruby[:1500] + generic[:500]

    all_data = []  # list of (fused_hs, token_ids, target_logits)
    for i, text in enumerate(texts):
        ids = tok(text, return_tensors="pt", max_length=128, truncation=True)["input_ids"].to("cuda:0")
        if ids.size(1) < 10: continue
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
            # Fuse first 3 hidden states (EAGLE-3 approach)
            hs0 = out.hidden_states[0]  # embedding output
            hs1 = out.hidden_states[1]  # layer 1 output
            hs2 = out.hidden_states[2]  # layer 2 output
            fused = torch.cat([hs0, hs1, hs2], dim=-1).cpu()  # (1, seq, 3*hidden)
            target_logits = out.logits.cpu()  # (1, seq, vocab)
        all_data.append((fused.squeeze(0), ids.squeeze(0).cpu(), target_logits.squeeze(0)))
        if (i+1) % 500 == 0:
            print("  %d/%d" % (i+1, len(texts)))
    print("Collected %d sequences" % len(all_data))

    # Free model layers
    del model.model.layers
    import gc; gc.collect(); torch.cuda.empty_cache()

    # Build EAGLE-3 head
    head = EAGLE3Head(
        hidden_dim=config.hidden_size,
        n_heads=config.num_attention_heads,
        n_kv_heads=config.num_key_value_heads,
        ffn_dim=config.intermediate_size,
    ).to("cuda:0", dtype=torch.bfloat16)

    n_params = sum(p.numel() for p in head.parameters())
    print("EAGLE-3 head: %.1fM params" % (n_params / 1e6))

    # Training
    k_steps = 7
    n_epochs = 20
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.01)
    total_steps = len(all_data) * n_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    print("\nTraining (k=%d, %d epochs, %d sequences)..." % (k_steps, n_epochs, len(all_data)))
    t0 = time.time()
    best_acc = 0
    step = 0

    for epoch in range(n_epochs):
        indices = torch.randperm(len(all_data))
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0

        for idx in indices:
            fused_hs, token_ids, target_logits = all_data[idx]
            fused_hs = fused_hs.to("cuda:0")
            token_ids = token_ids.to("cuda:0")
            target_logits = target_logits.to("cuda:0")
            seq_len = token_ids.size(0)

            if seq_len < k_steps + 2: continue

            # Random start position
            max_start = seq_len - k_steps - 1
            start = torch.randint(0, max(1, max_start), (1,)).item()

            # Fuse the starting hidden state
            fused_start = head.fc(fused_hs[start:start+1].unsqueeze(0).to(torch.bfloat16))  # (1, 1, hidden_dim)

            # Autoregressive chain
            h = fused_start
            kv = None
            total_loss = torch.tensor(0.0, device="cuda:0")
            n_toks = 0

            for j in range(k_steps):
                pos = start + j
                if pos + 1 >= seq_len: break

                # Token embedding (teacher-forced: use real token)
                tok_emb = embed(token_ids[pos:pos+1]).unsqueeze(0).to(torch.bfloat16)  # (1, 1, hidden_dim)

                # Draft step
                h, h_normed, kv = head(tok_emb, h, kv)

                # Logits from LM head
                logits = lm_head(h_normed).squeeze(0).float()  # (1, vocab)

                # Soft target: KL divergence against target model distribution
                target_pos = pos + 1
                target_dist = F.softmax(target_logits[target_pos:target_pos+1].float(), dim=-1)
                log_pred = F.log_softmax(logits, dim=-1)
                kl = F.kl_div(log_pred, target_dist, reduction='batchmean')
                total_loss = total_loss + kl
                n_toks += 1

                # Track accuracy
                with torch.no_grad():
                    pred_tok = logits.argmax(dim=-1).item()
                    real_tok = target_logits[target_pos].argmax(dim=-1).item()
                    if pred_tok == real_tok:
                        epoch_correct += 1
                    epoch_total += 1

            if n_toks == 0: continue
            avg_loss = total_loss / n_toks

            optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += avg_loss.item()
            step += 1

            if step % 500 == 0:
                acc = epoch_correct / max(epoch_total, 1)
                elapsed = time.time() - t0
                if acc > best_acc:
                    best_acc = acc
                    torch.save(head.state_dict(), "/root/t6b-mogae/eagle3_head_best.pt")
                    tag = " *BEST*"
                else:
                    tag = ""
                print("  Step %d (ep%d): loss=%.3f acc=%.1f%% best=%.1f%% (%.0fs)%s" % (
                    step, epoch, epoch_loss / step, acc * 100, best_acc * 100, elapsed, tag))

        acc = epoch_correct / max(epoch_total, 1)
        print("  Epoch %d done: acc=%.1f%%" % (epoch, acc * 100))

    torch.save(head.state_dict(), "/root/t6b-mogae/eagle3_head_final.pt")
    print("\nDone in %.0fs. Best: %.1f%%" % (time.time()-t0, best_acc*100))


if __name__ == "__main__":
    main()
