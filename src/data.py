"""Data loading, patient-grouped splitting, prompt construction and target encoding.

Only the TrialGPT Criterion Annotations dataset is used.  The columns that hold
GPT-4 outputs (gpt4_explanation, gpt4_sentences, gpt4_eligibility,
explanation_correctness) are dropped immediately after loading so that they can
never reach a prompt, a feature, or an evaluation routine.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PARQUET = ROOT / "data" / "raw" / "data" / "train-00000-of-00001.parquet"
PROCESSED = ROOT / "data" / "processed"

SEED = 42
N_TRAIN, N_VAL, N_TEST = 37, 8, 8

# Columns produced by GPT-4 in the original corpus.  They are removed on load.
LEAKY_COLUMNS = ["gpt4_explanation", "explanation_correctness", "gpt4_sentences", "gpt4_eligibility"]

INCLUSION_LABELS = ["included", "not included", "not enough information", "not applicable"]
EXCLUSION_LABELS = ["excluded", "not excluded", "not enough information", "not applicable"]
ALL_LABELS = ["included", "not included", "excluded", "not excluded",
              "not enough information", "not applicable"]
LABELS_FOR_TYPE = {"inclusion": INCLUSION_LABELS, "exclusion": EXCLUSION_LABELS}

_SENT_RE = re.compile(r"^\s*(\d+)\.\s*(.*)$")


def load_raw() -> pd.DataFrame:
    """Load the parquet file and strip every GPT-4 derived column."""
    df = pd.read_parquet(RAW_PARQUET)
    df = df.drop(columns=LEAKY_COLUMNS)
    # The upstream ``training`` flag is a different (original TrialGPT) partition;
    # it is deliberately ignored in favour of the assignment's own split.
    df = df.drop(columns=["training"])
    df["expert_sentences"] = df["expert_sentences"].apply(parse_id_list)
    return df


def parse_id_list(value) -> list[int]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(v) for v in value]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    value = str(value).strip()
    if not value:
        return []
    return sorted({int(v) for v in ast.literal_eval(value)})


def split_sentences(note: str) -> dict[int, str]:
    """Return {sentence_id: text} for a note that uses the '<id>. <text>' layout."""
    out: dict[int, str] = {}
    current = None
    for line in note.splitlines():
        m = _SENT_RE.match(line)
        if m:
            current = int(m.group(1))
            out[current] = m.group(2).strip()
        elif current is not None and line.strip():
            out[current] += " " + line.strip()  # continuation of the previous sentence
    return out


def make_split(patient_ids) -> dict[str, list[str]]:
    """Patient-grouped split: sorted ids -> default_rng(42).shuffle -> 37 / 8 / 8."""
    ids = np.array(sorted(set(patient_ids)))
    rng = np.random.default_rng(SEED)
    rng.shuffle(ids)
    assert len(ids) == N_TRAIN + N_VAL + N_TEST, f"expected 53 patients, got {len(ids)}"
    return {
        "train": ids[:N_TRAIN].tolist(),
        "validation": ids[N_TRAIN:N_TRAIN + N_VAL].tolist(),
        "test": ids[N_TRAIN + N_VAL:].tolist(),
    }


def check_no_leakage(df: pd.DataFrame, split: dict[str, list[str]]) -> dict:
    sets = {k: set(v) for k, v in split.items()}
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = sets[names[i]] & sets[names[j]]
            assert not overlap, f"patient leakage between {names[i]} and {names[j]}: {overlap}"
    covered = set().union(*sets.values())
    assert covered == set(df.patient_id.unique()), "some patients are not assigned to a split"
    # every annotation of a patient must be in exactly one split
    counts = {k: int(df.patient_id.isin(v).sum()) for k, v in sets.items()}
    assert sum(counts.values()) == len(df)
    return counts


def build_manifest() -> dict:
    df = load_raw()
    split = make_split(df.patient_id.unique())
    counts = check_no_leakage(df, split)
    manifest = {
        "seed": SEED,
        "procedure": "sorted unique patient ids -> numpy.random.default_rng(42).shuffle -> first 37 train, next 8 validation, last 8 test",
        "n_patients": {k: len(v) for k, v in split.items()},
        "n_annotations": counts,
        "n_usable_annotations": {k: int(df[df.patient_id.isin(v) & df.criterion_text.notna()].shape[0])
                                 for k, v in split.items()},
        "excluded_annotation_ids": df[df.criterion_text.isna()].annotation_id.tolist(),
        "excluded_reason": "criterion_text is missing in the source dataset",
        "patients": split,
    }
    PROCESSED.mkdir(parents=True, exist_ok=True)
    (PROCESSED / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_split(name: str) -> pd.DataFrame:
    manifest = json.loads((PROCESSED / "split_manifest.json").read_text())
    df = load_raw()
    df = df[df.patient_id.isin(manifest["patients"][name])]
    # annotation 883 has no criterion text at all -> nothing to predict; excluded (documented)
    df = df[df.criterion_text.notna()]
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Prompt / target format
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a clinical-trial eligibility screening assistant. Given a numbered patient note "
    "and one trial criterion, decide the criterion-level label and list the IDs of the note "
    "sentences that support it. Answer only with JSON."
)


def build_user_message(note: str, criterion_type: str, criterion_text: str) -> str:
    labels = ", ".join(LABELS_FOR_TYPE[criterion_type])
    return (
        f"Patient note (each sentence is numbered):\n{note.strip()}\n\n"
        f"Criterion type: {criterion_type}\n"
        f"Criterion: {criterion_text.strip()}\n\n"
        f"Allowed labels: {labels}.\n"
        'Reply as {"label": <label>, "evidence": [<sentence ids>]}.'
    )


def build_messages(note: str, criterion_type: str, criterion_text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(note, criterion_type, criterion_text)},
    ]


def encode_target(label: str, evidence: list[int]) -> str:
    return json.dumps({"label": label, "evidence": sorted(evidence)})


_JSON_RE = re.compile(r"\{.*?\}", re.S)


def parse_output(text: str, criterion_type: str) -> tuple[str, list[int], bool]:
    """Parse model text -> (label, evidence ids, ok).  Invalid output gets label 'invalid'."""
    m = _JSON_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            label = str(obj.get("label", "")).strip().lower()
            ev = [int(x) for x in obj.get("evidence", [])]
            if label in LABELS_FOR_TYPE[criterion_type]:
                return label, sorted(set(ev)), True
            return "invalid", sorted(set(ev)), False
        except Exception:
            pass
    # lenient fall-back: look for a label string and bracketed ids
    low = text.lower()
    found = [lab for lab in sorted(LABELS_FOR_TYPE[criterion_type], key=len, reverse=True) if lab in low]
    ev_m = re.search(r"\[([\d,\s]*)\]", text)
    ev = [int(x) for x in re.findall(r"\d+", ev_m.group(1))] if ev_m else []
    if found:
        return found[0], sorted(set(ev)), False
    return "invalid", sorted(set(ev)), False


if __name__ == "__main__":
    mf = build_manifest()
    print(json.dumps({k: mf[k] for k in ("seed", "n_patients", "n_annotations")}, indent=2))
