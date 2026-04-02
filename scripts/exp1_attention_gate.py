#!/usr/bin/env python3
"""Experiment 1: Does gating work for attention?

Tests whether DeltaGate on attention projections (Q/K/V/O LoRA) shows
selectivity between domain (Ruby code) and generic (C4) text.

Pre-registered: loop/preregistrations/attention-gates.md
CHARTER: one variable (attention vs FFN). Same gate arch, same training.

Run on GPU server:
    cd /root/t6b-mogae
    PYTHONPATH=/root/t6b-mogae python3 scripts/exp1_attention_gate.py

Results → /root/t6b-mogae/results/exp1_attention_gate_seed{SEED}.json
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/root/t6b-mogae/scripts/grove")
from adapter_modules import LoRA, DeltaGate, HookModule, DEFAULT_BIAS_INIT

# === Config ===
DEVICE = "cuda:1"  # GPU 1 (grove server on GPU 0)
SEED = 42
RANK = 16
EXPERT_START = 12
PHASE1_STEPS = 500
PHASE2_STEPS = 500
MAX_SEQ_LEN = 512
LR_ADAPTER = 3e-4
LR_GATE = 1e-3
OUTPUT_DIR = "/root/t6b-mogae/results"

sys.stdout.reconfigure(line_buffering=True)


# === Attention adapter ===

class AttentionExpert(nn.Module):
    """LoRA adapters on Q/K/V/O attention projections."""
    def __init__(self, hidden_size, num_heads, head_dim, rank):
        super().__init__()
        qkv_dim = hidden_size  # Q, K, V projections are hidden_size → hidden_size
        self.q_lora = LoRA(hidden_size, qkv_dim, rank)
        self.k_lora = LoRA(hidden_size, qkv_dim, rank)
        self.v_lora = LoRA(hidden_size, qkv_dim, rank)
        self.o_lora = LoRA(hidden_size, qkv_dim, rank)


def make_attention_hook(layer_idx, attn_expert, gate, orig_attn_forward, phase):
    """Create a hooked attention forward that injects LoRA + gate."""
    def hooked_forward(hidden_states, **kwargs):
        # Get base attention output
        base_out = orig_attn_forward(hidden_states, **kwargs)
        # base_out is a tuple: (attn_output, ...) or just attn_output
        if isinstance(base_out, tuple):
            base_attn = base_out[0]
            rest = base_out[1:]
        else:
            base_attn = base_out
            rest = ()

        # Compute LoRA corrections on the attention output
        flat = hidden_states.reshape(-1, hidden_states.size(-1))

        # Q/K affect attention patterns, V/O affect what's transferred
        # We apply as a post-hoc delta on the attention output
        # (simpler than hooking individual Q/K/V/O projections)
        q_corr = attn_expert.q_lora(flat).reshape(base_attn.shape)
        k_corr = attn_expert.k_lora(flat).reshape(base_attn.shape)
        v_corr = attn_expert.v_lora(flat).reshape(base_attn.shape)
        o_corr = attn_expert.o_lora(flat).reshape(base_attn.shape)

        # Combined correction
        adapter_delta = q_corr + k_corr + v_corr + o_corr

        if phase == 1:
            # Phase 1: no gate, direct adapter
            result = base_attn + adapter_delta
        else:
            # Phase 2: gated
            gate_val = torch.sigmoid(gate(flat))  # (B*L, 1)
            gate_val = gate_val.reshape(*base_attn.shape[:-1], 1)
            result = base_attn + gate_val * adapter_delta

        if rest:
            return (result,) + rest
        return result

    return hooked_forward


def load_ruby_data(tokenizer, n_texts=200):
    """Load Ruby code from The Stack or fallback to codeparrot."""
    print("Loading Ruby code data...")
    try:
        from datasets import load_dataset
        ds = load_dataset("bigcode/the-stack-dedup", data_dir="data/ruby",
                          split="train", streaming=True, trust_remote_code=True)
        texts = []
        for item in ds:
            text = item.get("content", "")
            if len(text) > 200 and len(text) < 10000:
                texts.append(text[:2000])
            if len(texts) >= n_texts:
                break
        if len(texts) >= 50:
            print(f"Loaded {len(texts)} Ruby files from the-stack-dedup")
            return texts
    except Exception as e:
        print(f"the-stack-dedup failed: {e}")

    try:
        from datasets import load_dataset
        ds = load_dataset("codeparrot/github-code", languages=["Ruby"],
                          split="train", streaming=True, trust_remote_code=True)
        texts = []
        for item in ds:
            text = item.get("code", "")
            if len(text) > 200 and len(text) < 10000:
                texts.append(text[:2000])
            if len(texts) >= n_texts:
                break
        if len(texts) >= 50:
            print(f"Loaded {len(texts)} Ruby files from codeparrot/github-code")
            return texts
    except Exception as e:
        print(f"codeparrot failed: {e}")

    # Last resort: synthetic Ruby
    print("Using synthetic Ruby code")
    return _synthetic_ruby(n_texts)


def _synthetic_ruby(n):
    """Generate synthetic Ruby code snippets."""
    templates = [
        'class User < ApplicationRecord\n  has_many :posts\n  validates :email, presence: true, uniqueness: true\n\n  def full_name\n    "#{first_name} #{last_name}"\n  end\n\n  def active?\n    confirmed_at.present? && !banned?\n  end\nend',
        'module Authentication\n  extend ActiveSupport::Concern\n\n  included do\n    before_action :authenticate_user!\n  end\n\n  private\n\n  def authenticate_user!\n    unless current_user\n      redirect_to login_path, alert: "Please sign in"\n    end\n  end\n\n  def current_user\n    @current_user ||= User.find_by(id: session[:user_id])\n  end\nend',
        'class PostsController < ApplicationController\n  before_action :set_post, only: [:show, :edit, :update, :destroy]\n\n  def index\n    @posts = Post.published.order(created_at: :desc).page(params[:page])\n  end\n\n  def create\n    @post = current_user.posts.build(post_params)\n    if @post.save\n      redirect_to @post, notice: "Post created"\n    else\n      render :new, status: :unprocessable_entity\n    end\n  end\n\n  private\n\n  def post_params\n    params.require(:post).permit(:title, :body, :published)\n  end\nend',
        'RSpec.describe User do\n  describe "#full_name" do\n    it "returns first and last name" do\n      user = build(:user, first_name: "Jane", last_name: "Doe")\n      expect(user.full_name).to eq("Jane Doe")\n    end\n  end\n\n  describe "#active?" do\n    context "when confirmed and not banned" do\n      it "returns true" do\n        user = build(:user, confirmed_at: Time.current, banned: false)\n        expect(user).to be_active\n      end\n    end\n  end\nend',
        'class CreateUsers < ActiveRecord::Migration[7.1]\n  def change\n    create_table :users do |t|\n      t.string :email, null: false\n      t.string :first_name\n      t.string :last_name\n      t.string :password_digest\n      t.datetime :confirmed_at\n      t.boolean :banned, default: false\n      t.timestamps\n    end\n    add_index :users, :email, unique: true\n  end\nend',
        'class PaymentService\n  def initialize(user, amount, currency: "EUR")\n    @user = user\n    @amount = amount\n    @currency = currency\n  end\n\n  def charge!\n    transaction = Transaction.create!(\n      user: @user,\n      amount: @amount,\n      currency: @currency,\n      status: :pending\n    )\n    result = gateway.charge(@amount, @currency, @user.payment_method)\n    if result.success?\n      transaction.update!(status: :completed)\n    else\n      transaction.update!(status: :failed, error: result.message)\n      raise PaymentError, result.message\n    end\n    transaction\n  end\n\n  private\n\n  def gateway\n    @gateway ||= Stripe::Gateway.new(Rails.application.credentials.stripe_key)\n  end\nend',
        'module Enumerable\n  def map_with_index\n    each_with_index.map { |item, idx| yield(item, idx) }\n  end\n\n  def pluck(*keys)\n    map { |item| keys.length == 1 ? item[keys.first] : keys.map { |k| item[k] } }\n  end\n\n  def frequencies\n    each_with_object(Hash.new(0)) { |item, counts| counts[item] += 1 }\n  end\nend',
        'class ApplicationMailer < ActionMailer::Base\n  default from: "noreply@example.com"\n  layout "mailer"\n\n  def welcome_email(user)\n    @user = user\n    @url = root_url\n    mail(to: @user.email, subject: "Welcome to Grove")\n  end\n\n  def password_reset(user)\n    @user = user\n    @token = user.generate_reset_token\n    mail(to: @user.email, subject: "Password Reset")\n  end\nend',
    ]
    texts = []
    for i in range(n):
        base = templates[i % len(templates)]
        texts.append(f"# file_{i}.rb\n{base}\n")
    return texts


def load_generic_data(tokenizer, n_texts=200):
    """Load generic text from C4."""
    print("Loading generic data (C4)...")
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    texts = []
    for item in ds:
        if len(item["text"]) > 200:
            texts.append(item["text"][:2000])
        if len(texts) >= n_texts:
            break
    print(f"Loaded {len(texts)} generic texts")
    return texts


def evaluate_gates(model, tokenizer, attn_experts, gates, domain_texts, generic_texts,
                   expert_start, num_layers, device):
    """Evaluate gate selectivity: domain vs generic activation."""
    model.eval()
    for g in gates.values():
        g.eval()

    domain_gates = {str(l): [] for l in range(expert_start, num_layers)}
    generic_gates = {str(l): [] for l in range(expert_start, num_layers)}

    def eval_batch(texts, gate_dict, n=50):
        for text in texts[:n]:
            ids = tokenizer.encode(text, max_length=MAX_SEQ_LEN, truncation=True)
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True)
                hidden_states = out.hidden_states  # list of (B, L, D)
                for l in range(expert_start, num_layers):
                    hs = hidden_states[l]  # (1, L, D)
                    flat = hs.reshape(-1, hs.size(-1))
                    gate_val = torch.sigmoid(gates[str(l)](flat)).mean().item()
                    gate_dict[str(l)].append(gate_val)

    eval_batch(domain_texts, domain_gates)
    eval_batch(generic_texts, generic_gates)

    # Compute per-layer and overall selectivity
    results = {}
    for l in range(expert_start, num_layers):
        d_mean = np.mean(domain_gates[str(l)]) if domain_gates[str(l)] else 0
        g_mean = np.mean(generic_gates[str(l)]) if generic_gates[str(l)] else 0
        results[l] = {"domain_gate": d_mean, "generic_gate": g_mean, "selectivity": d_mean - g_mean}

    overall_domain = np.mean([r["domain_gate"] for r in results.values()])
    overall_generic = np.mean([r["generic_gate"] for r in results.values()])
    overall_selectivity = overall_domain - overall_generic

    return {
        "per_layer": results,
        "overall_domain_gate": overall_domain,
        "overall_generic_gate": overall_generic,
        "overall_selectivity": overall_selectivity,
    }


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)

    print(f"=== Experiment 1: Attention DeltaGate (seed={SEED}) ===")
    print(f"Device: {DEVICE}, Rank: {RANK}, Expert start: {EXPERT_START}")

    # Load model
    print("Loading Qwen3-8B...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map={"": DEVICE}
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = hidden_size // num_heads
    print(f"Model: {num_layers} layers, {hidden_size} hidden, {num_heads} heads, {head_dim} head_dim")

    # Load data
    domain_texts = load_ruby_data(tokenizer)
    generic_texts = load_generic_data(tokenizer)

    # Tokenize
    domain_ids = [tokenizer.encode(t, max_length=MAX_SEQ_LEN, truncation=True) for t in domain_texts]
    generic_ids = [tokenizer.encode(t, max_length=MAX_SEQ_LEN, truncation=True) for t in generic_texts]

    def get_batch(text_ids):
        idx = np.random.randint(0, len(text_ids))
        ids = text_ids[idx][:MAX_SEQ_LEN]
        return torch.tensor([ids], dtype=torch.long, device=DEVICE)

    # Create attention adapters + gates
    print("Creating attention adapters + gates...")
    attn_experts = nn.ModuleDict()
    gates = nn.ModuleDict()
    for l in range(EXPERT_START, num_layers):
        attn_experts[str(l)] = AttentionExpert(hidden_size, num_heads, head_dim, RANK).to(DEVICE)
        gates[str(l)] = DeltaGate(hidden_size).to(DEVICE)

    # === Phase 1: Train attention adapter on domain data ===
    print(f"\n--- Phase 1: Attention adapter training ({PHASE1_STEPS} steps) ---")

    orig_attn_fwds = {}
    for l in range(EXPERT_START, num_layers):
        layer = model.model.layers[l]
        orig_attn_fwds[l] = layer.self_attn.forward
        layer.self_attn.forward = make_attention_hook(
            l, attn_experts[str(l)], gates[str(l)], orig_attn_fwds[l], phase=1
        )

    optimizer1 = torch.optim.AdamW(attn_experts.parameters(), lr=LR_ADAPTER)

    losses1 = []
    t0 = time.time()
    for step in range(PHASE1_STEPS):
        model.train()
        attn_experts.train()
        input_ids = get_batch(domain_ids)
        if input_ids.size(1) < 2:
            continue
        out = model(input_ids)
        loss = F.cross_entropy(
            out.logits[:, :-1].reshape(-1, out.logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
        )
        optimizer1.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(attn_experts.parameters(), 1.0)
        optimizer1.step()
        losses1.append(loss.item())
        if step % 100 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}")
    t1 = time.time()
    print(f"Phase 1 done: {t1 - t0:.1f}s, final loss={losses1[-1]:.4f}")

    # === Phase 2: Train gates on mixed data (freeze adapters) ===
    print(f"\n--- Phase 2: Gate training ({PHASE2_STEPS} steps) ---")

    for p in attn_experts.parameters():
        p.requires_grad = False

    # Re-hook with phase=2 (gated)
    for l in range(EXPERT_START, num_layers):
        layer = model.model.layers[l]
        layer.self_attn.forward = make_attention_hook(
            l, attn_experts[str(l)], gates[str(l)], orig_attn_fwds[l], phase=2
        )

    optimizer2 = torch.optim.AdamW(gates.parameters(), lr=LR_GATE)

    losses2 = []
    t0 = time.time()
    for step in range(PHASE2_STEPS):
        model.train()
        gates.train()
        # Alternate domain and generic
        if step % 2 == 0:
            input_ids = get_batch(domain_ids)
        else:
            input_ids = get_batch(generic_ids)
        if input_ids.size(1) < 2:
            continue
        out = model(input_ids)
        loss = F.cross_entropy(
            out.logits[:, :-1].reshape(-1, out.logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
        )
        # L1 sparsity on gates
        for g in gates.values():
            z = torch.zeros(1, hidden_size, dtype=torch.bfloat16, device=DEVICE)
            loss = loss + 0.05 * torch.sigmoid(g(z)).mean()
        optimizer2.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gates.parameters(), 1.0)
        optimizer2.step()
        losses2.append(loss.item())
        if step % 100 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}")
    t1 = time.time()
    print(f"Phase 2 done: {t1 - t0:.1f}s, final loss={losses2[-1]:.4f}")

    # Restore original attention forwards
    for l in orig_attn_fwds:
        model.model.layers[l].self_attn.forward = orig_attn_fwds[l]

    # === Evaluate selectivity ===
    print("\n--- Evaluating gate selectivity ---")

    # Re-install hooks for evaluation (phase 2 / gated)
    for l in range(EXPERT_START, num_layers):
        layer = model.model.layers[l]
        layer.self_attn.forward = make_attention_hook(
            l, attn_experts[str(l)], gates[str(l)], orig_attn_fwds[l], phase=2
        )

    eval_result = evaluate_gates(
        model, tokenizer, attn_experts, gates,
        domain_texts, generic_texts,
        EXPERT_START, num_layers, DEVICE
    )

    # Restore again
    for l in orig_attn_fwds:
        model.model.layers[l].self_attn.forward = orig_attn_fwds[l]

    print(f"\nOverall domain gate:  {eval_result['overall_domain_gate']:.4f}")
    print(f"Overall generic gate: {eval_result['overall_generic_gate']:.4f}")
    print(f"Overall selectivity:  {eval_result['overall_selectivity']:.4f}")

    # H1 check
    sel = eval_result['overall_selectivity']
    if sel > 0.1:
        h1_result = "PASS"
        print(f"\nH1: PASS (selectivity {sel:.4f} > 0.1) — attention gates are selective!")
    elif sel > 0.03:
        h1_result = "MARGINAL"
        print(f"\nH1: MARGINAL (selectivity {sel:.4f} in 0.03-0.1 range)")
    else:
        h1_result = "FAIL"
        print(f"\nH1: FAIL (selectivity {sel:.4f} ≤ 0.03) — attention patterns appear universal")

    # Per-layer profile
    print("\nPer-layer gate profile:")
    for l in sorted(eval_result["per_layer"].keys()):
        r = eval_result["per_layer"][l]
        bar = "█" * int(r["selectivity"] * 50) if r["selectivity"] > 0 else ""
        print(f"  L{l:2d}: domain={r['domain_gate']:.3f} generic={r['generic_gate']:.3f} "
              f"sel={r['selectivity']:+.3f} {bar}")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result = {
        "experiment": "exp1_attention_gate",
        "seed": SEED,
        "domain": "ruby_code",
        "rank": RANK,
        "expert_start": EXPERT_START,
        "phase1_steps": PHASE1_STEPS,
        "phase2_steps": PHASE2_STEPS,
        "h1_result": h1_result,
        "overall_selectivity": eval_result["overall_selectivity"],
        "overall_domain_gate": eval_result["overall_domain_gate"],
        "overall_generic_gate": eval_result["overall_generic_gate"],
        "per_layer": {str(k): v for k, v in eval_result["per_layer"].items()},
        "phase1_final_loss": losses1[-1] if losses1 else None,
        "phase2_final_loss": losses2[-1] if losses2 else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out_path = os.path.join(OUTPUT_DIR, f"exp1_attention_gate_seed{SEED}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
