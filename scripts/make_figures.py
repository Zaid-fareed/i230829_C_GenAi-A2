"""Produce every figure used in the paper from the saved logs / metric files (no hand-entered numbers)."""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = os.environ.get("RUN", "cpu")
PFX = "gpu_" if RUN == "gpu" else ""
M = ROOT / "results" / ("metrics_gpu" if RUN == "gpu" else "metrics")
MC = ROOT / "results" / "metrics"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
(ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)

C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7", "grey": "#666666"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "axes.grid": True, "grid.alpha": .25, "legend.frameon": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{PFX}{name}.pdf"); fig.savefig(ROOT / "results" / "figures" / f"{PFX}{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", name)


def jload(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


# ---------------------------------------------------------------- training curves
def training():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    runs = ([("train_360m.csv", "smollm2-360m-lora", "Target: SmolLM2-360M (CPU)"), ("train_135m.csv", "smollm2-135m-lora", "Draft: SmolLM2-135M (CPU)")]
            if RUN == "cpu" else
            [("train_1.7b_gpu.csv", "smollm2-1.7b-lora", "Target: SmolLM2-1.7B (GPU)"), ("train_360m_gpu.csv", "smollm2-360m-gpu-lora", "Draft: SmolLM2-360M (GPU)")])
    for ax, (csv, folder, title) in zip(axes, runs):
        p = ROOT / "results" / "logs" / csv
        if not p.exists():
            continue
        d = pd.read_csv(p)
        ax.plot(d.step, d.train_loss, color=C["blue"], alpha=.35, lw=1, label="train loss (step)")
        ax.plot(d.step, d.train_loss.rolling(5, min_periods=1).mean(), color=C["blue"], lw=1.8, label="train loss (5-step mean)")
        h = jload(ROOT / "checkpoints" / "adapters" / folder / "epoch_summary.json") or []
        if h:
            ax.plot([e["step"] for e in h], [e["val_loss"] for e in h], "o-", color=C["red"], lw=1.5, label="validation loss (epoch)")
            ax2 = ax.twinx(); ax2.grid(False); ax2.spines["right"].set_visible(True)
            ax2.plot([e["step"] for e in h], [e["val_macro_f1"] for e in h], "s--", color=C["green"], lw=1.2, label="val macro-F1")
            ax2.set_ylabel("val macro-F1"); ax2.set_ylim(0, 1)
        ax.set_xlabel("optimizer step"); ax.set_ylabel("answer-token loss"); ax.set_title(title); ax.legend(loc="upper right", fontsize=6)
    save(fig, "fig_training")


# ---------------------------------------------------------------- confusion matrices
def confusion():
    r = jload(M / "finetuned_test.json")
    if not r:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, key in zip(axes, ["overall", "inclusion", "exclusion"]):
        c = r[key]["classification"]; cm = np.array(c["confusion_matrix"]); labs = [l.replace("not enough information", "NEI").replace("not applicable", "N/A") for l in c["labels"]]
        ax.imshow(cm, cmap="Blues"); ax.set_xticks(range(len(labs))); ax.set_yticks(range(len(labs)))
        ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=6); ax.set_yticklabels(labs, fontsize=6); ax.grid(False)
        for i in range(len(labs)):
            for j in range(len(labs)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6, color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_title(f"{key} (acc {c['accuracy']:.2f}, F1 {c['macro_f1']:.2f})", fontsize=7); ax.set_xlabel("predicted"); ax.set_ylabel("expert")
    save(fig, "fig_confusion")


# ---------------------------------------------------------------- before / after
def before_after():
    a, b = jload(M / "base_zeroshot_test.json"), jload(M / "finetuned_test.json")
    mj = jload(MC / "baseline_majority.json")
    if not (a and b):
        return
    names = ["accuracy", "macro-F1", "evidence\nmicro-F1"]
    get = lambda r: [r["overall"]["classification"]["accuracy"], r["overall"]["classification"]["macro_f1"], r["overall"]["evidence"]["micro"]["f1"]]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = np.arange(3); w = .26
    series = [("base, zero-shot", get(a), C["grey"]), ("fine-tuned", get(b), C["blue"])]
    if mj:
        series.insert(0, ("majority baseline", get(mj["test"]), C["orange"]))
    w = .8 / len(series)
    for k, (n, v, col) in enumerate(series):
        bars = ax.bar(x + (k - (len(series) - 1) / 2) * w, v, w, label=n, color=col)
        for rect, val in zip(bars, v):
            ax.text(rect.get_x() + rect.get_width() / 2, val + .01, f"{val:.2f}", ha="center", fontsize=5.5)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 1.05); ax.legend(fontsize=6); ax.set_title("Held-out test split")
    save(fig, "fig_before_after")


# ---------------------------------------------------------------- KV cache
def kv():
    g = jload(M / "bench_kv.json")
    if not g:
        return
    df = pd.DataFrame(g)
    ins = sorted(df.n_in.unique())
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    col = {"uncached": C["red"], "kv_cache": C["blue"], "kv_cache_tiled": C["green"]}
    for ax, n_in in zip(axes, ins):
        for m, sub in df[df.n_in == n_in].groupby("mode"):
            sub = sub.sort_values("n_out"); ax.plot(sub.n_out, sub.total_s, "o-", color=col[m], label=m, lw=1.4)
        ax.set_title(f"prompt = {n_in} tokens"); ax.set_xlabel("generated tokens"); ax.set_ylabel("total time (s)")
        ax.set_yscale("log")
    axes[0].legend(fontsize=6)
    save(fig, "fig_kv_time")
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for m, sub in df[df.n_out == df.n_out.max()].groupby("mode"):
        sub = sub.sort_values("n_in"); ax.plot(sub.n_in, sub.peak_delta_mb, "o-", color=col[m], label=m)
    ax.set_xlabel("prompt tokens"); ax.set_ylabel("peak memory growth (MB)"); ax.set_title(f"{df.n_out.max()} new tokens"); ax.legend(fontsize=6)
    save(fig, "fig_kv_memory")


# ---------------------------------------------------------------- attention
def attn():
    g = jload(M / "bench_attention.json")
    if not g:
        return
    df = pd.DataFrame(g)
    col = {"reference": C["red"], "tiled": C["green"], "sdpa": C["blue"]}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    pre = df[(df.kind == "prefill") & ((df.impl != "tiled") | (df.block == 64))]
    for m, sub in pre.groupby("impl"):
        sub = sub.sort_values("T")
        axes[0].plot(sub["T"], sub.median_ms, "o-", color=col[m], label=m)
        axes[1].plot(sub["T"], sub.peak_delta_mb.clip(lower=0.5), "o-", color=col[m], label=m)
        if m != "reference":
            axes[2].plot(sub["T"], sub.max_abs_err.clip(lower=1e-9), "o-", color=col[m], label=m)
    axes[0].set_yscale("log"); axes[1].set_yscale("log"); axes[2].set_yscale("log")
    axes[0].set_ylabel("median time (ms)"); axes[1].set_ylabel("peak memory growth (MB)"); axes[2].set_ylabel("max |error| vs reference")
    for a, t in zip(axes, ["runtime", "memory", "numerical error"]):
        a.set_xlabel("sequence length T"); a.set_title(f"prefill {t}"); a.set_xscale("log", base=2)
    axes[0].legend(fontsize=6)
    save(fig, "fig_attention")
    b = df[(df.impl == "tiled") & (df.kind == "prefill") & (df["T"] == 1024)].sort_values("block")
    if len(b) > 1:
        fig, ax = plt.subplots(figsize=(3.2, 2.2))
        ax.plot(b.block, b.median_ms, "o-", color=C["green"]); ax.set_xscale("log", base=2)
        ax.set_xlabel("tile size (rows = columns)"); ax.set_ylabel("median time (ms), T=1024"); ax.set_title("Tiled attention: tile-size sweep")
        save(fig, "fig_tile_sweep")


# ---------------------------------------------------------------- speculative
def spec():
    r = jload(M / "bench_spec.json")
    if not r:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for row, (setting, s) in enumerate(r["settings"].items()):
        df = pd.DataFrame(s["speculative"])
        for dn, sub in df.groupby("draft"):
            c = C["blue"] if dn == "finetuned_draft" else C["orange"]
            ls = "o-" if setting == "natural_output" else "s--"
            if setting == "natural_output":
                axes[0].plot(sub.k, 100 * sub.acceptance_rate, ls, color=c, label=dn.replace("_", " "))
            axes[1 if setting == "natural_output" else 2].plot(sub.k, sub.speedup_vs_greedy, ls, color=c, label=dn.replace("_", " "))
    axes[0].set_ylabel("draft acceptance rate (%)"); axes[0].set_title("natural task output")
    axes[1].set_ylabel("speed-up vs greedy target"); axes[1].set_title("natural task output"); axes[1].axhline(1, color="k", lw=.8)
    axes[2].set_ylabel("speed-up vs greedy target"); axes[2].set_title("forced 64-token output"); axes[2].axhline(1, color="k", lw=.8)
    for a in axes:
        a.set_xlabel("draft length k")
    axes[0].legend(fontsize=6)
    save(fig, "fig_speculative")


if __name__ == "__main__":
    for f in (training, confusion, before_after, kv, attn, spec):
        try:
            f()
        except Exception as e:  # keep going so one missing result does not block the rest
            print("FAILED", f.__name__, repr(e), file=sys.stderr)
