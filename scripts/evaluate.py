"""Evaluate a model on a split with greedy decoding; writes predictions + metrics.

Examples
  python scripts/evaluate.py --split validation --adapter none   --tag base_zeroshot
  python scripts/evaluate.py --split test       --adapter best   --tag finetuned
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data as D  # noqa: E402
import metrics as M  # noqa: E402
from pipeline import Engine, TARGET_ADAPTER, RESULT_TAG  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["validation", "test"])
    ap.add_argument("--adapter", default="best", help="'none', 'best', or a path to an adapter dir")
    ap.add_argument("--mode", default="kv_cache")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    adapter = None if args.adapter == "none" else (TARGET_ADAPTER if args.adapter == "best" else Path(args.adapter))
    eng = Engine(target_adapter=adapter, load_draft=(args.mode == "speculative"))
    df = D.load_split(args.split)
    if args.limit:
        df = df.head(args.limit)
    out_dir = ROOT / "results" / RESULT_TAG
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, preds = [], []
    t0 = time.time()
    for n, (_, r) in enumerate(df.iterrows()):
        p = eng.predict(r["note"], r["criterion_type"], r["criterion_text"], mode=args.mode)
        rows.append({"criterion_type": r["criterion_type"], "gold_label": r["expert_eligibility"],
                     "pred_label": p["label"], "gold_ev": r["expert_sentences"], "pred_ev": p["evidence_ids"]})
        preds.append({"annotation_id": int(r["annotation_id"]), "patient_id": r["patient_id"], "trial_id": r["trial_id"],
                      "criterion_type": r["criterion_type"], "criterion_text": r["criterion_text"],
                      "gold_label": r["expert_eligibility"], "gold_evidence": r["expert_sentences"],
                      "pred_label": p["label"], "pred_evidence": p["evidence_ids"], "raw_output": p["raw_output"],
                      "well_formed": p["output_well_formed"], "latency_s": p["latency_s"],
                      "n_prompt_tokens": p["n_prompt_tokens"], "n_new_tokens": p["n_new_tokens"]})
        if (n + 1) % 20 == 0:
            print(f"{n + 1}/{len(df)}  {time.time() - t0:.0f}s", flush=True)
    rep = M.full_report(rows)
    rep["meta"] = {"tag": args.tag, "split": args.split, "adapter": str(adapter), "mode": args.mode,
                   "decoding": "greedy (argmax), max_new_tokens=48, fp32", "n": len(rows),
                   "well_formed_fraction": float(np.mean([p["well_formed"] for p in preds])),
                   "eval_seconds": time.time() - t0}
    (out_dir / f"{args.tag}_{args.split}.json").write_text(json.dumps(rep, indent=2))
    with open(out_dir / f"{args.tag}_{args.split}_predictions.jsonl", "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    c = rep["overall"]["classification"]; e = rep["overall"]["evidence"]["micro"]
    print(f"[{args.tag}/{args.split}] acc={c['accuracy']:.3f} macroF1={c['macro_f1']:.3f} "
          f"evidence microF1={e['f1']:.3f} invalid={c['n_invalid']}")


if __name__ == "__main__":
    main()
