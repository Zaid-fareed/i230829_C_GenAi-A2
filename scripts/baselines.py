"""Trivial reference baselines (no model): majority label per criterion type learned on TRAIN only,
and empty evidence.  They show how much of the accuracy comes from class imbalance alone."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data as D  # noqa: E402
import metrics as M  # noqa: E402

train = D.load_split("train")
maj = {t: Counter(train[train.criterion_type == t].expert_eligibility).most_common(1)[0][0] for t in ("inclusion", "exclusion")}
print("majority labels (from train):", maj)
out = {"majority_labels": maj}
for split in ("validation", "test"):
    df = D.load_split(split)
    rows = [{"criterion_type": r.criterion_type, "gold_label": r.expert_eligibility, "pred_label": maj[r.criterion_type],
             "gold_ev": r.expert_sentences, "pred_ev": []} for r in df.itertuples()]
    rep = M.full_report(rows)
    out[split] = rep
    c = rep["overall"]["classification"]
    print(split, f"acc={c['accuracy']:.3f} macroF1={c['macro_f1']:.3f}")
(ROOT / "results" / "metrics").mkdir(parents=True, exist_ok=True)
(ROOT / "results" / "metrics" / "baseline_majority.json").write_text(json.dumps(out, indent=2))
