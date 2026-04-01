"""Verify GraphableDecodeStep matches HF native logits on a real model.

Quick correctness test that can run on the GPU server.
Usage: PYTHONPATH=/root/t6b-mogae python verify_fix.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from grove_server.engine.static_kv_cache import StaticKVCache
from grove_server.engine.graphable_model import GraphableDecodeStep


def main():
    model_name = "Qwen/Qwen3-0.6B"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    print(f"Loading {model_name} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    # Check if model has q_norm/k_norm
    attn0 = model.model.layers[0].self_attn
    has_qk_norm = hasattr(attn0, "q_norm")
    print(f"Model has QK norms: {has_qk_norm}")

    prompt = "The quick brown fox"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.size(1)

    # --- HF native forward ---
    with torch.no_grad():
        hf_out = model(input_ids)
        hf_logits = hf_out.logits  # (1, seq_len, vocab_size)

    # --- Our GraphableDecodeStep (token by token) ---
    config = model.config
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    max_seq_len = 2048

    cache = StaticKVCache(
        num_layers=config.num_hidden_layers,
        num_heads=num_kv_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        batch_size=1,
        dtype=dtype,
        device=device,
    )
    step = GraphableDecodeStep(model, cache, max_seq_len=max_seq_len)

    our_logits_list = []
    with torch.no_grad():
        for pos in range(seq_len):
            token = input_ids[:, pos:pos+1]
            position_ids = torch.tensor([[pos]], device=device)
            logits = step(token, position_ids)
            our_logits_list.append(logits)

    # Compare last token logits (most meaningful for generation)
    our_last = our_logits_list[-1][0, 0]  # (vocab_size,)
    hf_last = hf_logits[0, -1]  # (vocab_size,)

    max_diff = (our_last.float() - hf_last.float()).abs().max().item()
    our_top5 = our_last.topk(5).indices.tolist()
    hf_top5 = hf_last.topk(5).indices.tolist()

    print(f"\nMax logits diff (last token): {max_diff:.4f}")
    print(f"HF top5:  {hf_top5}")
    print(f"Our top5: {our_top5}")
    print(f"Top5 match: {our_top5 == hf_top5}")

    # Compute PPL for a longer sequence
    import math
    text = "The quick brown fox jumps over the lazy dog and runs through the forest"
    enc = tokenizer(text, return_tensors="pt").to(device)
    ids = enc["input_ids"]

    # HF PPL
    with torch.no_grad():
        hf_out2 = model(ids)
        hf_logits2 = hf_out2.logits
    shift_logits = hf_logits2[:, :-1, :].float()
    shift_labels = ids[:, 1:]
    loss_hf = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
    )
    ppl_hf = math.exp(loss_hf.item())

    # Our PPL
    cache2 = StaticKVCache(
        num_layers=config.num_hidden_layers,
        num_heads=num_kv_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        batch_size=1,
        dtype=dtype,
        device=device,
    )
    step2 = GraphableDecodeStep(model, cache2, max_seq_len=max_seq_len)
    our_all_logits = []
    with torch.no_grad():
        for pos in range(ids.size(1)):
            token = ids[:, pos:pos+1]
            position_ids = torch.tensor([[pos]], device=device)
            logits = step2(token, position_ids)
            our_all_logits.append(logits[0, 0])

    our_logits_stack = torch.stack(our_all_logits[:-1], dim=0).float()  # (seq-1, vocab)
    labels = ids[0, 1:]
    loss_ours = torch.nn.functional.cross_entropy(our_logits_stack, labels)
    ppl_ours = math.exp(loss_ours.item())

    print(f"\nHF native PPL:  {ppl_hf:.2f}")
    print(f"Our PPL:        {ppl_ours:.2f}")
    print(f"PPL ratio:      {ppl_ours/ppl_hf:.4f}")

    if max_diff < 0.1 and abs(ppl_ours - ppl_hf) / ppl_hf < 0.01:
        print("\nCORRECTNESS VERIFIED")
    elif max_diff < 1.0:
        print("\nWARNING: small numerical differences (likely BF16 accumulation)")
    else:
        print(f"\nFAILED: logits diverge significantly (max_diff={max_diff:.2f})")


if __name__ == "__main__":
    main()
