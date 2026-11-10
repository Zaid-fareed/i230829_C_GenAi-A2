"""Architecture diagrams for the paper (drawn with matplotlib so they are reproducible from code).

fig_system.pdf   : end-to-end pipeline and application workflow
fig_decoder.pdf  : our Llama-style decoder with explicit KV cache, pluggable attention and speculative loop
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUTS = [ROOT / "report" / "figures", ROOT / "results" / "figures"]
BLUE, ORANGE, GREEN, GREY, PURPLE = "#dbe8f7", "#fde7c9", "#dcf2e4", "#eeeeee", "#efe0f5"
plt.rcParams.update({"font.size": 7})


def box(ax, x, y, w, h, text, color, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02", fc=color, ec="#444444", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=5.4, weight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, text=None, dashed=False):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, lw=0.9, color="#333333",
                                 linestyle="--" if dashed else "-"))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.018, text, ha="center", va="bottom", fontsize=5.8, color="#333333")


def save(fig, name):
    for d in OUTS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.{'pdf' if d.parent.name == 'report' else 'png'}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ system / workflow
fig, ax = plt.subplots(figsize=(8.6, 3.2)); ax.set_xlim(-0.01, 1.02); ax.set_ylim(-0.01, 1); ax.axis("off")
# training lane
ax.text(0.01, 0.95, "Training (offline)", fontsize=7.5, weight="bold")
box(ax, 0.01, 0.70, 0.15, 0.18, "TrialGPT criterion\nannotations\n(GPT-4 columns dropped)", GREY)
box(ax, 0.20, 0.70, 0.15, 0.18, "Patient-grouped split\nseed 42: 37 / 8 / 8\n+ leakage checks", GREY)
box(ax, 0.39, 0.70, 0.17, 0.18, "Prompt: numbered note\n+ criterion type + text\nTarget: {label, evidence}", BLUE)
box(ax, 0.60, 0.70, 0.17, 0.18, "LoRA fine-tuning\n(answer-token loss)\nbest epoch by val macro-F1", ORANGE)
box(ax, 0.81, 0.70, 0.18, 0.18, "Adapters: 1.7B / 360M\n(target), 360M / 135M\n(draft)", ORANGE)
for a, b in ((0.16, 0.20), (0.35, 0.39), (0.56, 0.60), (0.77, 0.81)):
    arrow(ax, a, 0.79, b, 0.79)
# inference lane
ax.text(0.01, 0.56, "Inference / browser application", fontsize=7.5, weight="bold")
box(ax, 0.01, 0.30, 0.14, 0.20, "User input:\npatient note +\ntrial criteria\n(sample, NCT id,\npasted, manual)", GREY)
box(ax, 0.18, 0.30, 0.13, 0.20, "Validate input,\nnumber sentences,\nsplit criteria", GREY)
box(ax, 0.34, 0.30, 0.20, 0.20, "Own decoder (MiniLlama)\nLoRA merged W+(a/r)BA\nmode: uncached | KV cache |\nKV + tiled | speculative", GREEN, bold=False)
box(ax, 0.57, 0.30, 0.13, 0.20, "Parse JSON:\nlabel + sentence\nIDs (invalid ->\nflagged)", BLUE)
box(ax, 0.73, 0.30, 0.12, 0.20, "Evidence text +\nextractive\nexplanation", BLUE)
box(ax, 0.88, 0.30, 0.11, 0.20, "Rule-based\npreliminary\noutcome", PURPLE)
for a, b in ((0.15, 0.18), (0.31, 0.34), (0.54, 0.57), (0.70, 0.73), (0.85, 0.88)):
    arrow(ax, a, 0.40, b, 0.40)
arrow(ax, 0.90, 0.70, 0.46, 0.50, "load + merge adapter", dashed=True)
box(ax, 0.34, 0.03, 0.20, 0.17, "Compare-decoding tab:\nsame input, all modes,\ntokens match? latency,\ntok/s, memory, acceptance", GREEN)
box(ax, 0.73, 0.03, 0.26, 0.17, "Report export (HTML / Markdown / JSON)\nwith provenance: model, adapter hash,\ndecoding settings, time; disclaimer", PURPLE)
arrow(ax, 0.44, 0.30, 0.44, 0.20)
arrow(ax, 0.935, 0.30, 0.86, 0.20)
save(fig, "fig_system")

# ------------------------------------------------------------------ decoder + cache + speculative
fig, ax = plt.subplots(figsize=(8.6, 3.4)); ax.set_xlim(-0.01, 1.02); ax.set_ylim(-0.01, 1); ax.axis("off")
ax.text(0.01, 0.95, "(a) One decoder step with the explicit KV cache", fontsize=7.5, weight="bold")
box(ax, 0.01, 0.74, 0.12, 0.14, "new tokens\n(T of them)\npositions L..L+T-1", GREY)
box(ax, 0.16, 0.74, 0.10, 0.14, "embedding", BLUE)
box(ax, 0.29, 0.74, 0.11, 0.14, "RMSNorm\nQ, K, V proj.", BLUE)
box(ax, 0.43, 0.74, 0.10, 0.14, "RoPE on Q, K\n(absolute pos.)", BLUE)
box(ax, 0.56, 0.74, 0.14, 0.14, "attention backend:\nreference | tiled\nonline-softmax | SDPA\nmask: key j <= L+i", GREEN)
box(ax, 0.73, 0.74, 0.11, 0.14, "out proj. +\nresidual,\nRMSNorm, SwiGLU", BLUE)
box(ax, 0.87, 0.74, 0.12, 0.14, "x N layers ->\nnorm -> LM head\n-> logits", BLUE)
for a, b in ((0.13, 0.16), (0.26, 0.29), (0.40, 0.43), (0.53, 0.56), (0.70, 0.73), (0.84, 0.87)):
    arrow(ax, a, 0.81, b, 0.81)
box(ax, 0.40, 0.50, 0.33, 0.14, "KVCache per layer: K, V buffers (B, H_kv, capacity, d)\nwrite rotated K,V at [L, L+T) ; capacity doubles when full\nlength += T after the step ; truncate(n) = rollback", ORANGE)
arrow(ax, 0.45, 0.74, 0.47, 0.64); ax.text(0.40, 0.675, "write K,V", fontsize=5.6)
arrow(ax, 0.63, 0.64, 0.63, 0.74); ax.text(0.645, 0.675, "read [0, L+T)", fontsize=5.6)
ax.text(0.01, 0.40, "(b) Greedy speculative decoding round (target T, draft D share the tokenizer)", fontsize=7.5, weight="bold")
box(ax, 0.01, 0.08, 0.16, 0.24, "Invariant: both caches\nhold every accepted\ntoken except 'last'", GREY)
box(ax, 0.21, 0.08, 0.16, 0.24, "D drafts k tokens\nd1..dk greedily from\n'last' (k cheap passes)", PURPLE)
box(ax, 0.41, 0.08, 0.17, 0.24, "T verifies in ONE pass\non [last, d1..dk]\n-> greedy g0..gk", GREEN)
box(ax, 0.62, 0.08, 0.17, 0.24, "accept longest prefix\nd(i+1) = g(i); emit\nd1..dn + g(n)\n(n+1 >= 1 tokens)", BLUE)
box(ax, 0.83, 0.08, 0.16, 0.24, "rollback: truncate T\nand D caches; if n=k\nfeed dk to D;\nlast <- g(n)", ORANGE)
for a, b in ((0.17, 0.21), (0.37, 0.41), (0.58, 0.62), (0.79, 0.83)):
    arrow(ax, a, 0.20, b, 0.20)
arrow(ax, 0.91, 0.08, 0.09, 0.08, dashed=True)
ax.text(0.50, 0.015, "next round  (output is identical to greedy decoding of T)", ha="center", fontsize=5.8)
save(fig, "fig_decoder")
