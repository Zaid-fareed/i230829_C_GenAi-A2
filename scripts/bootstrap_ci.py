"""Patient-level bootstrap confidence intervals (rows of one patient are resampled together).

usage: python scripts/bootstrap_ci.py finetuned test
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import metrics as M  # noqa: E402

tag, split = sys.argv[1], sys.argv[2]
MD = "metrics_gpu" if os.environ.get("RUN") == "gpu" else "metrics"
B = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
preds = [json.loads(l) for l in open(ROOT / "results" / MD / f"{tag}_{split}_predictions.jsonl")]
by_pat = {}
for p in preds:
    by_pat.setdefault(p["patient_id"], []).append(p)
pats = sorted(by_pat)
rng = np.random.default_rng(42)
accs, f1s, evs = [], [], []
for _ in range(B):
    sample = [r for pid in rng.choice(pats, size=len(pats), replace=True) for r in by_pat[pid]]
    rows = [{"criterion_type": r["criterion_type"], "gold_label": r["gold_label"], "pred_label": r["pred_label"],
             "gold_ev": r["gold_evidence"], "pred_ev": r["pred_evidence"]} for r in sample]
    c = M.classification_report([r["gold_label"] for r in rows], [r["pred_label"] for r in rows])
    accs.append(c["accuracy"]); f1s.append(c["macro_f1"])
    evs.append(M.evidence_report([r["gold_ev"] for r in rows], [r["pred_ev"] for r in rows])["micro"]["f1"])
ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
out = {"tag": tag, "split": split, "n_patients": len(pats), "bootstrap_samples": B,
       "accuracy_95ci": ci(accs), "macro_f1_95ci": ci(f1s), "evidence_micro_f1_95ci": ci(evs)}
(ROOT / "results" / MD / f"{tag}_{split}_bootstrap.json").write_text(json.dumps(out, indent=2))
print(out)
