"""Generate LaTeX tables and \\newcommand number macros from result files -> report/generated/*.tex"""
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data as D  # noqa: E402

OUT = ROOT / "report" / "generated"
OUT.mkdir(parents=True, exist_ok=True)
RUN = os.environ.get("RUN", "cpu")                     # cpu | gpu
PFX = "gpu_" if RUN == "gpu" else ""                    # file-name prefix for generated tables
MP = "G" if RUN == "gpu" else ""                        # LaTeX macro prefix
M = ROOT / "results" / ("metrics_gpu" if RUN == "gpu" else "metrics")
MC = ROOT / "results" / "metrics"                       # model-free / CPU-only files (baseline, candidates)


def J(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def tex(s):
    s = str(s)
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_"),
                 ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def f(x, n=3):
    return "--" if x is None else f"{x:.{n}f}"


def cut(t):
    suf = "\\midrule\n"
    return t[: -len(suf)] if t.endswith(suf) else t


def write(name, body):
    (OUT / (PFX + name)).write_text(body, encoding="utf-8")
    print("wrote", name)


NUM = {}


def num(k, v):
    NUM[MP + k] = v


SHORT = {"not enough information": "not enough info.", "not applicable": "not applicable"}

# ------------------------------------------------------------- split / label table
man = J(ROOT / "data/processed/split_manifest.json")
if man:
    rows = []
    for sp in ("train", "validation", "test"):
        df = D.load_split(sp)
        rows.append((sp, len(man["patients"][sp]), len(df), (df.criterion_type == "inclusion").sum(), (df.criterion_type == "exclusion").sum()))
    t = "\\begin{tabular}{lrrrr}\\toprule\nSplit & Patients & Annotations & Incl. & Excl. \\\\\\midrule\n"
    t += "".join(f"{a} & {b} & {c} & {d} & {e} \\\\\n" for a, b, c, d, e in rows) + "\\bottomrule\\end{tabular}\n"
    write("tab_split.tex", t)
    labs = D.ALL_LABELS
    t = "\\begin{tabular}{lrrr}\\toprule\nLabel & Train & Val. & Test \\\\\\midrule\n"
    dfs = {sp: D.load_split(sp) for sp in ("train", "validation", "test")}
    for l in labs:
        t += f"{tex(SHORT.get(l, l))} & " + " & ".join(str((dfs[sp].expert_eligibility == l).sum()) for sp in dfs) + " \\\\\n"
    write("tab_labels.tex", t + "\\bottomrule\\end{tabular}\n")
    num("NTrain", rows[0][2]); num("NVal", rows[1][2]); num("NTest", rows[2][2])

# ------------------------------------------------------------- candidates
cc = J(MC / "candidate_compare.json") if RUN == "cpu" else None
if cc:
    t = "\\begin{tabular}{lrrrrrr}\\toprule\nModel & Params (M) & fp32 (MB) & Vocab & Ctx & LoRA step (s)$^\\dagger$ & Zero-shot JSON \\\\\\midrule\n"
    for n, r in cc.items():
        t += f"{tex(n)} & {r['parameters_M']:.0f} & {r['fp32_weights_MB']:.0f} & {r['vocab']} & {r['max_position_embeddings']} & {r['lora_step_batch4_seconds']:.1f} & {tex(r['zero_shot_well_formed_json'])} \\\\\n"
    write("tab_candidates.tex", t + "\\bottomrule\\end{tabular}\n")

# ------------------------------------------------------------- training summary
RUNS = ((("target", "T", "smollm2-360m-lora"), ("draft", "D", "smollm2-135m-lora")) if RUN == "cpu"
        else (("target", "T", "smollm2-1.7b-lora"), ("draft", "D", "smollm2-360m-gpu-lora")))
for tag, key, folder in RUNS:
    d = ROOT / "checkpoints" / "adapters" / folder
    h, b, s = J(d / "epoch_summary.json"), J(d / "best.json"), J(d / "data_stats.json")
    if h:
        t = "\\begin{tabular}{rrrrr}\\toprule\nEpoch & Val. loss & Val. acc. & Val. macro-F1 & Val. evid. F1 \\\\\\midrule\n"
        for e in h:
            star = "$^*$" if b and e["epoch"] == b["best_epoch"] else ""
            t += f"{e['epoch']}{star} & {e['val_loss']:.3f} & {e['val_accuracy']:.3f} & {e['val_macro_f1']:.3f} & {e['val_evidence_micro_f1']:.3f} \\\\\n"
        write(f"tab_epochs_{tag}.tex", t + "\\bottomrule\\end{tabular}\n")
        num(f"{key}BestEpoch", b["best_epoch"] if b else "?"); num(f"{key}TrainHours", f"{h[-1]['elapsed_s'] / 3600:.1f}")
    if s:
        num(f"{key}MaxTok", s["max_tokens_seen"]); num(f"{key}MeanTok", f"{s['mean_tokens']:.0f}"); num(f"{key}Trunc", s["n_truncated"])

# ------------------------------------------------------------- before / after + per-class
base_t, fine_t = J(M / "base_zeroshot_test.json"), J(M / "finetuned_test.json")
base_v, fine_v = J(M / "base_zeroshot_validation.json"), J(M / "finetuned_validation.json")
maj = J(MC / "baseline_majority.json")
if fine_t and base_t:
    def line(name, r):
        o, i, e = r["overall"], r["inclusion"], r["exclusion"]
        return (f"{name} & {f(o['classification']['accuracy'])} & {f(o['classification']['macro_f1'])} & {f(i['classification']['accuracy'])} & {f(i['classification']['macro_f1'])} & "
                f"{f(e['classification']['accuracy'])} & {f(e['classification']['macro_f1'])} & {o['classification']['n_invalid']} \\\\\n")
    t = "\\begin{tabular}{llrrrrrrr}\\toprule\nSplit & Model & Acc. & mF1 & Incl. acc. & Incl. mF1 & Excl. acc. & Excl. mF1 & Invalid \\\\\\midrule\n"
    blocks = []
    for split, key, a, b in (("Val.", "validation", base_v, fine_v), ("Test", "test", base_t, fine_t)):
        if not (a and b):
            continue
        lines = []
        if maj:
            lines.append(line("majority label", maj[key]))
        lines += [line("base zero-shot", a), line("fine-tuned", b)]
        lines[0] = f"{split} & " + lines[0]
        lines[1:] = ["& " + l for l in lines[1:]]
        blocks.append("".join(lines))
    write("tab_main.tex", t + "\\midrule\n".join(blocks) + "\\bottomrule\\end{tabular}\n")
    c = fine_t["overall"]["classification"]; ev = fine_t["overall"]["evidence"]
    num("TestAcc", f(c["accuracy"])); num("TestMF", f(c["macro_f1"])); num("TestEvF", f(ev["micro"]["f1"]))
    num("BaseAcc", f(base_t["overall"]["classification"]["accuracy"])); num("BaseMF", f(base_t["overall"]["classification"]["macro_f1"]))
    num("BaseEvF", f(base_t["overall"]["evidence"]["micro"]["f1"]))
    num("BaseInvalid", base_t["overall"]["classification"]["n_invalid"]); num("TestInvalid", c["n_invalid"])
    if maj:
        num("MajAcc", f(maj["test"]["overall"]["classification"]["accuracy"])); num("MajMF", f(maj["test"]["overall"]["classification"]["macro_f1"]))
    # per-class
    for name, r in (("finetuned", fine_t), ("base", base_t)):
        t = "\\begin{tabular}{llrrrr}\\toprule\nScope & Class & P & R & F1 & Support \\\\\\midrule\n"
        for scope in ("inclusion", "exclusion"):
            pc = r[scope]["classification"]["per_class"]
            for k, (l, v) in enumerate(pc.items()):
                t += f"{scope if k == 0 else ''} & {tex(SHORT.get(l, l))} & {f(v['precision'])} & {f(v['recall'])} & {f(v['f1'])} & {v['support']} \\\\\n"
            t += "\\midrule\n"
        write(f"tab_perclass_{name}.tex", cut(t) + "\\bottomrule\\end{tabular}\n")
    # evidence
    t = "\\begin{tabular}{llrrrr}\\toprule\nModel & Scope & P & R & F1 & Type \\\\\\midrule\n"
    for name, r in (("base", base_t), ("fine-tuned", fine_t)):
        for scope in ("overall", "inclusion", "exclusion"):
            e = r[scope]["evidence"]
            t += f"{name} & {scope} & {f(e['micro']['precision'])} & {f(e['micro']['recall'])} & {f(e['micro']['f1'])} & micro \\\\\n"
        e = r["overall"]["evidence"]["macro_nonempty_gold"]
        t += f"{name} & overall & {f(e['precision'])} & {f(e['recall'])} & {f(e['f1'])} & macro (gold$\\neq\\emptyset$) \\\\\n"
        e = r["overall"]["evidence"]["macro_all_records"]
        t += f"{name} & overall & {f(e['precision'])} & {f(e['recall'])} & {f(e['f1'])} & macro (all records) \\\\\\midrule\n"
    write("tab_evidence.tex", cut(t) + "\\bottomrule\\end{tabular}\n")
    ee = fine_t["overall"]["evidence"]
    num("NGoldEmpty", ee["n_gold_empty"]); num("NGoldNon", ee["n_gold_nonempty"]); num("EmptyOK", f(ee["empty_gold_correct"]))
    num("EmptyFalseCit", ee["empty_gold_false_citations"])

# ------------------------------------------------------------- KV
kv = J(M / "bench_kv.json")
if kv:
    t = "\\begin{tabular}{rrlrrrrrc}\\toprule\nIn & Out & Mode & Prefill (s) & Decode (s) & tok/s & Peak $\\Delta$ (MB) & KV (MB) & = ref. \\\\\\midrule\n"
    last = None
    for r in kv:
        key = (r["n_in"], r["n_out"])
        t += f"{r['n_in'] if key != last else ''} & {r['n_out'] if key != last else ''} & {tex(r['mode'])} & {r['prefill_s']:.3f} & {r['decode_s']:.3f} & {r['tok_per_s']:.1f} & {r['peak_delta_mb']:.0f} & {r['kv_cache_mb']:.1f} & {'yes' if r['tokens_match_reference'] else 'NO'} \\\\\n"
        if key != last and last is not None:
            pass
        last = key
    write("tab_kv.tex", t + "\\bottomrule\\end{tabular}\n")
    num("KVAllMatch", f"{sum(r['tokens_match_reference'] for r in kv)}/{len(kv)}")
    big = [r for r in kv if r["n_in"] == max(x["n_in"] for x in kv) and r["n_out"] == max(x["n_out"] for x in kv)]
    u = next((r for r in big if r["mode"] == "uncached"), None); c = next((r for r in big if r["mode"] == "kv_cache"), None)
    if u and c:
        num("KVSpeedup", f"{u['total_s'] / c['total_s']:.1f}"); num("KVBigIn", u["n_in"]); num("KVBigOut", u["n_out"])

# ------------------------------------------------------------- attention
at = J(M / "bench_attention.json")
if at:
    t = "\\begin{tabular}{llrrrrr}\\toprule\nPhase & Impl. & $T$ & Time (ms) & Max err. & Mean err. & Peak $\\Delta$ (MB) \\\\\\midrule\n"
    for kind in ("prefill", "decode"):
        for r in [x for x in at if x["kind"] == kind and (x["impl"] != "tiled" or x["block"] == 64) and x["T"] in (256, 1024, 2048)]:
            t += f"{kind} & {r['impl']} & {r['T']} & {r['median_ms']:.2f} & {r['max_abs_err']:.1e} & {r['mean_abs_err']:.1e} & {r['peak_delta_mb']:.0f} \\\\\n"
        t += "\\midrule\n"
    write("tab_attn.tex", cut(t) + "\\bottomrule\\end{tabular}\n")
    worst = max(r["max_abs_err"] for r in at if r["impl"] != "reference")
    num("AttnWorstErr", f"{worst:.1e}")

# ------------------------------------------------------------- speculative
sp = J(M / "bench_spec.json")
if sp:
    for setting, s in sp["settings"].items():
        g = s["greedy_target"]
        t = f"\\begin{{tabular}}{{llrrrrr}}\\toprule\nDraft & $k$ & Equal & Accept. & Latency (s) & tok/s & Speed-up \\\\\\midrule\n"
        t += f"\\multicolumn{{2}}{{l}}{{greedy target}} & -- & -- & {g['latency_s']:.3f} & {g['tok_per_s']:.1f} & 1.00 \\\\\\midrule\n"
        for r in s["speculative"]:
            t += f"{tex(r['draft'].replace('_', ' '))} & {r['k']} & {r['equal_to_greedy']} & {100 * r['acceptance_rate']:.0f}\\% & {r['latency_s']:.3f} & {r['tok_per_s']:.1f} & {r['speedup_vs_greedy']:.2f} \\\\\n"
        write(f"tab_spec_{setting}.tex", t + "\\bottomrule\\end{tabular}\n")
    best = max((r for s in sp["settings"].values() for r in s["speculative"]), key=lambda r: r["speedup_vs_greedy"])
    num("SpecBestSpeedup", f"{best['speedup_vs_greedy']:.2f}"); num("SpecBestK", best["k"]); num("SpecBestDraft", best["draft"].replace("_", " "))
    num("SpecHF", tex(sp.get("hf_assisted_generation_check", "")))

# ------------------------------------------------------------- errors
ea = J(ROOT / "results" / f"error_analysis_{RUN}_finetuned_test.json")
if ea:
    t = "\\begin{tabular}{p{.05\\linewidth}p{.15\\linewidth}p{.32\\linewidth}p{.13\\linewidth}p{.13\\linewidth}}\\toprule\nID & Type & Criterion (abridged) & Expert & Model \\\\\\midrule\n"
    for i, r in enumerate(ea["examples"][:12], 1):
        crit = r["criterion_text"][:90] + ("…" if len(r["criterion_text"]) > 90 else "")
        t += f"E{i} & {tex(r['category'].replace('_', ' '))} & {tex(crit)} & {tex(SHORT.get(r['gold_label'], r['gold_label']))} {tex(r['gold_evidence'])} & {tex(SHORT.get(r['pred_label'], r['pred_label']))} {tex(r['pred_evidence'])} \\\\\n"
    write("tab_errors.tex", t + "\\bottomrule\\end{tabular}\n")
    cc_ = ea["category_counts"]
    t = "\\begin{tabular}{lr}\\toprule\nError category & Count \\\\\\midrule\n" + "".join(f"{tex(k.replace('_', ' '))} & {v} \\\\\n" for k, v in sorted(cc_.items(), key=lambda kv: -kv[1])) + "\\bottomrule\\end{tabular}\n"
    write("tab_error_cats.tex", t)
    num("NContradict", ea["n_contradictory_decisive_without_evidence"]); num("NLabelErr", ea["n_label_errors"])

# ------------------------------------------------------------- macros
body = "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in NUM.items())
write("numbers.tex", body)

# ------------------------------------------------------------- bootstrap CIs (appended to macros)
bt = J(M / "finetuned_test_bootstrap.json"); bb = J(M / "base_zeroshot_test_bootstrap.json")
extra = ""
BS = chr(92)
for nm, r in (("Test", bt), ("Base", bb)):
    if r:
        for k, lab in (("accuracy_95ci", "Acc"), ("macro_f1_95ci", "MF"), ("evidence_micro_f1_95ci", "EvF")):
            for suffix, v in (("Lo", r[k][0]), ("Hi", r[k][1])):
                extra += BS + "newcommand{" + BS + MP + nm + lab + suffix + "}{" + f"{v:.3f}" + "}" + chr(10)
if extra:
    with open(OUT / (PFX + "numbers.tex"), "a", encoding="utf-8") as fh:
        fh.write(extra)
    print("appended bootstrap macros")
