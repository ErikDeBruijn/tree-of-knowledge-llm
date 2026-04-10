#!/usr/bin/env python3
"""Collect hidden state training data for EAGLE-3 draft head.

Runs Qwen3-8B on training data, extracts:
- Last hidden state (input to LM head) at each token position
- Next token label (what the model would predict)

Saves as .pt file for efficient training.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
sys.path.insert(0, "/root/t6b-mogae")

import json
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map={"": 0})
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model.eval()

    hidden_dim = model.config.hidden_size  # 4096
    n_layers = model.config.num_hidden_layers  # 36

    # Load training data
    ruby_texts = []
    with open("/root/t6b-mogae/training_data/ruby_domain.jsonl") as f:
        for line in f:
            ruby_texts.append(json.loads(line)["text"])
    generic_texts = []
    with open("/root/t6b-mogae/training_data/generic.jsonl") as f:
        for line in f:
            generic_texts.append(json.loads(line)["text"])

    print(f"Data: {len(ruby_texts)} Ruby, {len(generic_texts)} generic")

    # Collect hidden states
    all_hidden = []  # last layer hidden states
    all_labels = []  # next token ids
    max_samples = 5000  # enough for draft head training
    max_seq_len = 256
    target_tokens = 500_000

    total_tokens = 0
    t0 = time.time()

    # Mix domain and generic
    texts = ruby_texts[:max_samples//2] + generic_texts[:max_samples//2]

    for i, text in enumerate(texts):
        if total_tokens >= target_tokens:
            break

        ids = tok(text, return_tensors="pt", max_length=max_seq_len,
                  truncation=True)["input_ids"].to("cuda:0")

        if ids.size(1) < 4:
            continue

        with torch.no_grad():
            out = model(ids, output_hidden_states=True)

        # Last hidden state: shape (1, seq_len, hidden_dim)
        last_hs = out.hidden_states[-1]  # after final layer norm

        # Labels: shifted by 1 (predict next token)
        # hidden_state[t] should predict token[t+1]
        hidden = last_hs[:, :-1, :].squeeze(0).cpu()  # (seq_len-1, hidden_dim)
        labels = ids[:, 1:].squeeze(0).cpu()  # (seq_len-1,)

        all_hidden.append(hidden)
        all_labels.append(labels)
        total_tokens += hidden.size(0)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(texts)}: {total_tokens} tokens ({elapsed:.0f}s)")

    # Concatenate
    H = torch.cat(all_hidden, dim=0)  # (total_tokens, hidden_dim)
    L = torch.cat(all_labels, dim=0)  # (total_tokens,)

    print(f"\nCollected: {H.shape[0]} tokens, {H.shape[1]} dim")
    print(f"H dtype: {H.dtype}, size: {H.nelement() * 2 / 1e9:.2f} GB")

    # Save
    out_path = Path("/root/t6b-mogae/eagle_train_data.pt")
    torch.save({"hidden_states": H, "labels": L}, str(out_path))
    print(f"Saved to {out_path}")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
