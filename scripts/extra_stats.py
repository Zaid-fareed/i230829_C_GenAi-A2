"""Extra counts used in the paper's error discussion (contradictory / missing-information cases)."""
import json, os, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
MD = "metrics_gpu" if os.environ.get("RUN") == "gpu" else "metrics"
P = [json.loads(l) for l in open(ROOT / "results" / MD / "finetuned_test_predictions.jsonl")]
DEC = {"included", "not included", "excluded", "not excluded"}
out = {}
unc = [p for p in P if p["pred_label"] in DEC and not p["pred_evidence"]]
out["decisive_uncited"] = len(unc)
out["decisive_uncited_gold_has_evidence"] = sum(bool(p["gold_evidence"]) for p in unc)
out["decisive_uncited_gold_empty"] = sum(not p["gold_evidence"] for p in unc)
fd = [p for p in P if p["gold_label"] in ("not enough information", "not applicable") and p["pred_label"] in DEC]
out["false_decisions"] = len(fd); out["false_decisions_pred_labels"] = dict(Counter(p["pred_label"] for p in fd))
out["false_decisions_gold_labels"] = dict(Counter(p["gold_label"] for p in fd))
out["false_decisions_by_type"] = dict(Counter(p["criterion_type"] for p in fd))
out["nei_cited_gold_empty"] = sum(bool(p["pred_label"] == "not enough information" and p["pred_evidence"] and not p["gold_evidence"]) for p in P)
out["exclusion_pred_dist"] = dict(Counter(p["pred_label"] for p in P if p["criterion_type"] == "exclusion"))
out["inclusion_pred_dist"] = dict(Counter(p["pred_label"] for p in P if p["criterion_type"] == "inclusion"))
out["label_right_evidence_wrong"] = sum(p["gold_label"] == p["pred_label"] and set(p["gold_evidence"]) != set(p["pred_evidence"]) for p in P)
out["correct_label_and_evidence"] = sum(p["gold_label"] == p["pred_label"] and set(p["gold_evidence"]) == set(p["pred_evidence"]) for p in P)
(ROOT / "results" / MD / "finetuned_test_extra_stats.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
