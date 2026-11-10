"""Candidate-model table computed ONLY from each model's config.json (no training or timing involved).

Parameter count = embeddings + per-layer attention/MLP/norm weights (+ separate LM head if untied, + QKV biases for Qwen2).
Writes report/generated/tab_candidates.tex and results/metrics/candidate_table.json.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "SmolLM2-1.7B": ROOT / "docs/model_configs/SmolLM2-1.7B-Instruct/config.json",
    "SmolLM2-360M": ROOT / "checkpoints/base/SmolLM2-360M-Instruct/config.json",
    "SmolLM2-135M": ROOT / "checkpoints/base/SmolLM2-135M-Instruct/config.json",
    "Qwen2.5-0.5B": ROOT / "checkpoints/base/Qwen2.5-0.5B-Instruct/config.json",
}
ROLE = {"SmolLM2-1.7B": "target (A)", "SmolLM2-360M": "target (B) / draft (A)", "SmolLM2-135M": "draft (B)",
        "Qwen2.5-0.5B": "rejected"}


def n_params(c):
    h, i, L, v = c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"], c["vocab_size"]
    nh, nkv = c["num_attention_heads"], c.get("num_key_value_heads", c["num_attention_heads"])
    hd = h // nh
    attn = h * h + 2 * h * (nkv * hd) + h * h                 # q, k, v, o
    bias = (h + 2 * nkv * hd) if c.get("model_type") == "qwen2" else 0
    mlp = 3 * h * i                                            # gate, up, down
    norms = 2 * h
    total = v * h + L * (attn + bias + mlp + norms) + h
    if not c.get("tie_word_embeddings", False):
        total += v * h
    return total


rows, out = [], {}
for name, p in MODELS.items():
    c = json.loads(p.read_text())
    n = n_params(c)
    kv_bytes = 2 * c["num_hidden_layers"] * c.get("num_key_value_heads", c["num_attention_heads"]) * (c["hidden_size"] // c["num_attention_heads"]) * 4
    out[name] = {"params_M": n / 1e6, "fp32_weights_GB": n * 4 / 1e9, "layers": c["num_hidden_layers"], "hidden": c["hidden_size"],
                 "heads": c["num_attention_heads"], "kv_heads": c.get("num_key_value_heads", c["num_attention_heads"]),
                 "vocab": c["vocab_size"], "context": c["max_position_embeddings"], "kv_bytes_per_token_fp32": kv_bytes,
                 "license": "Apache-2.0", "role": ROLE[name]}
    rows.append((name, out[name]))

t = ("\\begin{tabular}{lrrrrrrrl}\\toprule\n"
     "Model & Params & fp32 (GB) & Layers & Heads/KV & Vocab & Ctx & KV/token (kB) & Role \\\\\\midrule\n")
def thou(x):
    return f"{x:,}".replace(",", "\\,")        # LaTeX thin space as thousands separator


for name, r in rows:
    pm = f"{r['params_M'] / 1000:.2f}\\,B" if r["params_M"] > 1000 else f"{r['params_M']:.0f}\\,M"
    t += (f"{name} & {pm} & {r['fp32_weights_GB']:.2f} & {r['layers']} & {r['heads']}/{r['kv_heads']} & "
          f"{thou(r['vocab'])} & {thou(r['context'])} & {r['kv_bytes_per_token_fp32'] / 1000:.0f} & {r['role']} \\\\\n")
t += "\\bottomrule\\end{tabular}\n"
(ROOT / "report/generated/tab_candidates.tex").write_text(t, encoding="utf-8")
(ROOT / "results/metrics/candidate_table.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(t)
