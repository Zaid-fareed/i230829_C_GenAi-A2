"""Categorise test-set errors of the fine-tuned model and dump representative cases.

Categories (mutually exclusive, first match wins):
  malformed_output       output could not be parsed into a valid label for the criterion type
  false_decision         gold = not enough information / not applicable, model committed to a decisive label
  missed_decision        gold decisive, model said not enough information / not applicable
  polarity_flip          decisive but opposite (included vs not included, excluded vs not excluded)
  na_confusion           confusion between not applicable and the other labels (not NEI)
  evidence_only          label right, cited sentences differ from the expert's
Also reports "contradictory" cases: decisive label with no cited sentence.
"""
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data as D  # noqa: E402

tag = sys.argv[1] if len(sys.argv) > 1 else "finetuned"
split = sys.argv[2] if len(sys.argv) > 2 else "test"
RUN = os.environ.get("RUN", "cpu")
MD = "metrics_gpu" if RUN == "gpu" else "metrics"
preds = [json.loads(l) for l in open(ROOT / "results" / MD / f"{tag}_{split}_predictions.jsonl")]
df = D.load_split(split).set_index("annotation_id")
DEC = {"included", "not included", "excluded", "not excluded"}


def category(p):
    g, q = p["gold_label"], p["pred_label"]
    if q == "invalid" or not p["well_formed"] and q == "invalid":
        return "malformed_output"
    if g != q:
        if g in ("not enough information", "not applicable") and q in DEC:
            return "false_decision"
        if g in DEC and q in ("not enough information", "not applicable"):
            return "missed_decision"
        if g in DEC and q in DEC:
            return "polarity_flip"
        return "na_confusion"
    if set(p["gold_evidence"]) != set(p["pred_evidence"]):
        return "evidence_only"
    return "correct"


rows = []
for p in preds:
    p["category"] = category(p)
    note = D.split_sentences(df.loc[p["annotation_id"], "note"])
    p["gold_evidence_text"] = {i: note.get(i, "") for i in p["gold_evidence"]}
    p["pred_evidence_text"] = {i: note.get(i, "") for i in p["pred_evidence"]}
    p["contradictory"] = p["pred_label"] in DEC and not p["pred_evidence"]
    rows.append(p)

cnt = Counter(r["category"] for r in rows)
print(cnt)
by = defaultdict(list)
for r in rows:
    if r["category"] not in ("correct",):
        by[r["category"]].append(r)
# representative selection: up to 3 per category, preferring distinct patients, total >= 8
chosen = []
for cat, lst in sorted(by.items(), key=lambda kv: -len(kv[1])):
    seen = set()
    for r in lst:
        if r["patient_id"] not in seen:
            chosen.append(r); seen.add(r["patient_id"])
        if len([c for c in chosen if c["category"] == cat]) >= 3:
            break
out = {"tag": tag, "split": split, "category_counts": cnt, "n_contradictory_decisive_without_evidence": sum(r["contradictory"] for r in rows),
       "n_label_errors": sum(r["gold_label"] != r["pred_label"] for r in rows), "examples": chosen}
(ROOT / "results" / f"error_analysis_{RUN}_{tag}_{split}.json").write_text(json.dumps(out, indent=2, default=str))
md = [f"# Error analysis - {tag} on {split}", "", f"Category counts: {dict(cnt)}", f"Decisive predictions with no cited sentence: {out['n_contradictory_decisive_without_evidence']}", ""]
for i, r in enumerate(chosen, 1):
    md += [f"## E{i} [{r['category']}] patient {r['patient_id']}, trial {r['trial_id']}",
           f"- criterion ({r['criterion_type']}): {r['criterion_text']}",
           f"- expert: **{r['gold_label']}**, evidence {r['gold_evidence']}  | model: **{r['pred_label']}**, evidence {r['pred_evidence']}",
           f"- expert evidence text: {r['gold_evidence_text']}", f"- model evidence text: {r['pred_evidence_text']}", f"- raw output: `{r['raw_output']}`", ""]
(ROOT / "results" / f"error_analysis_{RUN}_{tag}_{split}.md").write_text("\n".join(md))
print("examples:", len(chosen))
