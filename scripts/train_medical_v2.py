#!/usr/bin/env python3
"""Medical expert with real PubMed data + LoRA+ + keyword eval."""
import json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
sys.path.insert(0, "/root/t6b-mogae/scripts/grove")
from adapter_modules import HookModule
sys.stdout.reconfigure(line_buffering=True)

DEVICE = "cuda:1"
SEED = 42
RANK = 16
ALPHA = 32
ES = 12
MAX_STEPS = 3000
SEQ = 512
LR_A = 1e-4
LR_B = 1.6e-3
DROP = 0.1
EVAL_EVERY = 500


class LA(nn.Module):
    def __init__(self, i, o, r, a, d=0.1):
        super().__init__()
        self.A = nn.Parameter(torch.randn(i, r, dtype=torch.bfloat16) * 0.01)
        self.B = nn.Parameter(torch.zeros(r, o, dtype=torch.bfloat16))
        self.sc = a / r
        self.dr = nn.Dropout(d) if d > 0 else nn.Identity()

    def forward(self, x):
        return self.dr(x) @ self.A @ self.B * self.sc


class Exp(nn.Module):
    def __init__(self, h, i, r, a, d=0.1):
        super().__init__()
        self.gl = LA(h, i, r, a, d)
        self.ul = LA(h, i, r, a, d)

    def forward(self, x, m):
        return m.down_proj(F.silu(m.gate_proj(x) + self.gl(x)) * (m.up_proj(x) + self.ul(x)))


MED_EVAL = [
    {"q": "What is the first-line treatment for type 2 diabetes?", "kw": ["metformin"]},
    {"q": "What are the symptoms of myocardial infarction?", "kw": ["chest pain", "shortness of breath"]},
    {"q": "What is the normal range for blood pressure?", "kw": ["120", "80"]},
    {"q": "What antibiotic is used for strep throat?", "kw": ["penicillin", "amoxicillin"]},
    {"q": "What is the most common cause of pneumonia?", "kw": ["streptococcus", "pneumoniae"]},
    {"q": "What are common side effects of statins?", "kw": ["muscle", "liver"]},
    {"q": "What is the Glasgow Coma Scale used for?", "kw": ["consciousness", "brain", "injury"]},
    {"q": "What is the treatment for anaphylaxis?", "kw": ["epinephrine", "adrenaline"]},
    {"q": "What does HbA1c measure?", "kw": ["glucose", "blood sugar", "glycated"]},
    {"q": "What is the difference between Type 1 and Type 2 diabetes?", "kw": ["insulin", "autoimmune"]},
]


def med_eval(model, tok, device, label=""):
    hits = 0
    for mq in MED_EVAL:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": mq["q"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        ids = tok.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=200, do_sample=False, pad_token_id=tok.eos_token_id)
        answer = tok.decode(out[0][ids.size(1):], skip_special_tokens=True).lower()
        found = any(kw.lower() in answer for kw in mq["kw"])
        if found:
            hits += 1
        status = "HIT" if found else "MISS"
        print("  {} {:50s} {}".format(label, mq["q"][:50], status))
    rate = hits / len(MED_EVAL)
    print("  {} TOTAL: {}/{} ({:.0%})".format(label, hits, len(MED_EVAL), rate))
    return rate


def evaluate_ppl(model, tok, texts, device, n=100):
    model.eval()
    tl = tt = 0
    for t in texts[:n]:
        ids = tok.encode(t, max_length=SEQ, truncation=True)
        if len(ids) < 2:
            continue
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            o = model(x)
            l = F.cross_entropy(o.logits[:, :-1].reshape(-1, o.logits.size(-1)), x[:, 1:].reshape(-1), reduction="sum")
            tl += l.item()
            tt += x.size(1) - 1
    return torch.exp(torch.tensor(tl / tt)).item() if tt > 0 else float("inf")


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    print("=== Medical Expert (Real PubMed, LoRA+) ===")

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map={"": DEVICE})
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    NL = model.config.num_hidden_layers
    HS = model.config.hidden_size
    IS = model.config.intermediate_size

    print("Loading PubMed data...")
    ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    domain_texts = [
        str(item.get("question", "")) + " " + str(item.get("long_answer", ""))
        for item in ds if len(item.get("long_answer", "")) > 100
    ]
    np.random.shuffle(domain_texts)
    domain_texts = domain_texts[:3000]
    print("Domain: {} PubMed texts".format(len(domain_texts)))

    print("Loading generic data...")
    c4 = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    generic_texts = []
    for item in c4:
        if len(item["text"]) > 200:
            generic_texts.append(item["text"][:3000])
        if len(generic_texts) >= 2000:
            break

    domain_ids = [tok.encode(t, max_length=SEQ, truncation=True) for t in domain_texts]
    generic_ids = [tok.encode(t, max_length=SEQ, truncation=True) for t in generic_texts]

    print("\nBaseline:")
    base_dppl = evaluate_ppl(model, tok, domain_texts, DEVICE)
    base_gppl = evaluate_ppl(model, tok, generic_texts, DEVICE)
    print("  Domain PPL: {:.2f}, Generic PPL: {:.2f}".format(base_dppl, base_gppl))
    base_med = med_eval(model, tok, DEVICE, "BASE")

    ad = nn.ModuleDict()
    for l in range(ES, NL):
        ad[str(l)] = Exp(HS, IS, RANK, ALPHA, DROP).to(DEVICE)
    ap = []
    bp = []
    for l in range(ES, NL):
        ap.extend([ad[str(l)].gl.A, ad[str(l)].ul.A])
        bp.extend([ad[str(l)].gl.B, ad[str(l)].ul.B])
    opt = torch.optim.AdamW([{"params": ap, "lr": LR_A}, {"params": bp, "lr": LR_B}], weight_decay=0.05)
    wm = 300

    def lrf(step):
        if step < wm:
            return step / wm
        return max(0.1, 1.0 - (step - wm) / (MAX_STEPS - wm))

    sch = torch.optim.lr_scheduler.LambdaLR(opt, lrf)

    orig = {}
    for l in range(ES, NL):
        layer = model.model.layers[l]
        orig[l] = layer.mlp

        def mh(li, om):
            def h(hs):
                return ad[str(li)](hs.reshape(-1, hs.size(-1)), om).reshape(hs.shape)
            return HookModule(h)

        layer.mlp = mh(l, orig[l])

    best_med = base_med
    best_step = 0
    best_state = None
    history = []
    print("\nTraining {} steps:".format(MAX_STEPS))
    t0 = time.time()
    for step in range(MAX_STEPS):
        model.train()
        ad.train()
        if step % 5 < 4:
            idx = step % len(domain_ids)
            x = torch.tensor([domain_ids[idx][:SEQ]], dtype=torch.long, device=DEVICE)
        else:
            idx = np.random.randint(0, len(generic_ids))
            x = torch.tensor([generic_ids[idx][:SEQ]], dtype=torch.long, device=DEVICE)
        if x.size(1) < 2:
            continue
        loss = F.cross_entropy(model(x).logits[:, :-1].reshape(-1, model.config.vocab_size), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ap + bp, 1.0)
        opt.step()
        sch.step()

        if (step + 1) % EVAL_EVERY == 0:
            model.eval()
            ad.eval()
            mr = med_eval(model, tok, DEVICE, "Step {}".format(step + 1))
            print("    loss={:.4f} ({:.0f}s)".format(loss.item(), time.time() - t0))
            history.append({"step": step + 1, "med_rate": mr, "loss": loss.item()})
            if mr > best_med:
                best_med = mr
                best_step = step + 1
                best_state = {k: v.cpu().clone() for k, v in ad.state_dict().items()}
                print("    >>> NEW BEST <<<")

    for l in orig:
        model.model.layers[l].mlp = orig[l]

    sep = "=" * 60
    print("\n" + sep)
    print("  Base medical: {:.0%}".format(base_med))
    print("  Best medical: {:.0%} (step {})".format(best_med, best_step))
    for h in history:
        m = " <<<" if h["step"] == best_step else ""
        print("    Step {:5d}: med={:.0%} loss={:.4f}{}".format(h["step"], h["med_rate"], h["loss"], m))
    print(sep)

    os.makedirs("/root/t6b-mogae/results", exist_ok=True)
    with open("/root/t6b-mogae/results/medical_real_eval.json", "w") as f:
        json.dump({
            "base_med": base_med, "best_med": best_med, "best_step": best_step,
            "base_domain_ppl": base_dppl, "base_generic_ppl": base_gppl,
            "history": history, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, f, indent=2, default=str)
    print("Saved to results/medical_real_eval.json")


if __name__ == "__main__":
    main()
