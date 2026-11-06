"""Evaluation metrics for criterion classification and evidence-sentence retrieval."""
from __future__ import annotations

import numpy as np

from data import ALL_LABELS

INVALID = "invalid"


def confusion(y_true, y_pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    return cm


def classification_report(y_true, y_pred, labels=None):
    """Accuracy, macro-F1 and per-class P/R/F1.

    Predictions that are not valid labels (``invalid``) count as errors: they
    lower recall of the true class and are counted in the accuracy denominator,
    but they do not create a column in the confusion matrix.
    Macro-F1 averages over classes that occur in the gold labels *or* in the
    predictions (classes with zero support and zero predictions are skipped).
    """
    y_true, y_pred = list(y_true), list(y_pred)
    labels = labels or ALL_LABELS
    n = len(y_true)
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / max(n, 1)
    per = {}
    f1s = []
    for lab in labels:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred))
        support = tp + fn
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": support, "predicted": tp + fp}
        if support or (tp + fp):
            f1s.append(f1)
    macro = float(np.mean(f1s)) if f1s else 0.0
    return {
        "n": n,
        "accuracy": acc,
        "macro_f1": macro,
        "n_invalid": sum(p == INVALID for p in y_pred),
        "per_class": per,
        "confusion_matrix": confusion(y_true, y_pred, labels).tolist(),
        "labels": labels,
    }


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evidence_report(gold_list, pred_list):
    """Sentence-ID retrieval metrics.

    * micro   : TP/FP/FN pooled over every record (empty/empty records add nothing).
    * macro   : mean of per-record P/R/F1 over records whose gold set is NON-empty
                ("evidence-bearing" records); precision is 0 when nothing is predicted.
    * empty-gold policy: records with an empty gold set are scored separately by
                ``empty_gold_correct`` (fraction where the model also predicted an
                empty set) and ``empty_gold_false_citations`` (mean predicted ids).
                Absence of evidence is never converted to a label by this module.
    * all-record macro (``macro_all``) treats empty/empty as a perfect match (P=R=F1=1).
    """
    tp = fp = fn = 0
    rec_non, rec_all = [], []
    empty_total = empty_ok = 0
    empty_fp_ids = 0
    for g, p in zip(gold_list, pred_list):
        g, p = set(g), set(p)
        t, f_p, f_n = len(g & p), len(p - g), len(g - p)
        tp, fp, fn = tp + t, fp + f_p, fn + f_n
        if g:
            rec_non.append(_prf(t, f_p, f_n))
            rec_all.append(_prf(t, f_p, f_n))
        else:
            empty_total += 1
            if not p:
                empty_ok += 1
                rec_all.append((1.0, 1.0, 1.0))
            else:
                empty_fp_ids += len(p)
                rec_all.append((0.0, 0.0, 0.0))
    mp, mr, mf = _prf(tp, fp, fn)
    arr = np.array(rec_non) if rec_non else np.zeros((1, 3))
    arr_all = np.array(rec_all) if rec_all else np.zeros((1, 3))
    return {
        "n_records": len(gold_list),
        "n_gold_nonempty": len(rec_non),
        "n_gold_empty": empty_total,
        "micro": {"precision": mp, "recall": mr, "f1": mf, "tp": tp, "fp": fp, "fn": fn},
        "macro_nonempty_gold": {"precision": float(arr[:, 0].mean()), "recall": float(arr[:, 1].mean()),
                                "f1": float(arr[:, 2].mean())},
        "macro_all_records": {"precision": float(arr_all[:, 0].mean()), "recall": float(arr_all[:, 1].mean()),
                              "f1": float(arr_all[:, 2].mean())},
        "empty_gold_correct": empty_ok / empty_total if empty_total else None,
        "empty_gold_false_citations": empty_fp_ids,
    }


def full_report(rows):
    """rows: list of dicts with criterion_type, gold_label, pred_label, gold_ev, pred_ev."""
    out = {}
    for name, subset in (("overall", rows),
                         ("inclusion", [r for r in rows if r["criterion_type"] == "inclusion"]),
                         ("exclusion", [r for r in rows if r["criterion_type"] == "exclusion"])):
        if not subset:
            continue
        labels = ALL_LABELS if name == "overall" else (
            ["included", "not included", "not enough information", "not applicable"] if name == "inclusion"
            else ["excluded", "not excluded", "not enough information", "not applicable"])
        out[name] = {
            "classification": classification_report([r["gold_label"] for r in subset],
                                                    [r["pred_label"] for r in subset], labels),
            "evidence": evidence_report([r["gold_ev"] for r in subset], [r["pred_ev"] for r in subset]),
        }
    return out
