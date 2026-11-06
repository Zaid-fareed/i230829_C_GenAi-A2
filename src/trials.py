"""Helpers for turning trial eligibility text into individual criteria, and for the app's note handling."""
from __future__ import annotations

import re

import data as D

_HEAD_INC = re.compile(r"^\s*(?:key\s+)?inclusion\s+criteria\s*:?\s*$", re.I)
_HEAD_EXC = re.compile(r"^\s*(?:key\s+)?exclusion\s+criteria\s*:?\s*$", re.I)
_BULLET = re.compile(r"^\s*(?:[-*•●▪]+|\d{1,2}[.)])\s+")


def parse_criteria_text(text: str) -> list[dict]:
    """Split a free-text eligibility block ("Inclusion Criteria: ... Exclusion Criteria: ...") into items.

    Bullets (-, *, bullets, "1.") start a new criterion; unbulleted lines continue the previous one.
    Text before any heading is treated as inclusion.
    """
    section = "inclusion"
    items: list[dict] = []
    cur: list[str] | None = None

    def flush():
        nonlocal cur
        if cur:
            t = re.sub(r"\s+", " ", " ".join(cur)).strip()
            if len(t) >= 4:
                items.append({"type": section, "text": t})
        cur = None

    for raw in text.replace("\r", "").split("\n"):
        line = raw.rstrip()
        if _HEAD_INC.match(line):
            flush(); section = "inclusion"; continue
        if _HEAD_EXC.match(line):
            flush(); section = "exclusion"; continue
        if not line.strip():
            flush(); continue
        if _BULLET.match(line):
            flush(); cur = [_BULLET.sub("", line, count=1)]
        else:
            if cur is None:
                cur = [line.strip()]
            else:
                cur.append(line.strip())
    flush()
    return items


def fetch_clinicaltrials(nct_id: str, timeout=12) -> dict:
    """Fetch title + eligibility criteria from the public ClinicalTrials.gov v2 API (metadata only)."""
    import requests
    nct_id = nct_id.strip().upper()
    if not re.fullmatch(r"NCT\d{8}", nct_id):
        raise ValueError("A ClinicalTrials.gov id looks like NCT01234567")
    r = requests.get(f"https://clinicaltrials.gov/api/v2/studies/{nct_id}",
                     params={"fields": "NCTId,BriefTitle,EligibilityCriteria"}, timeout=timeout,
                     headers={"User-Agent": "educational-trial-screening-prototype"})
    if r.status_code == 404:
        raise ValueError(f"{nct_id} was not found on ClinicalTrials.gov")
    r.raise_for_status()
    p = r.json()["protocolSection"]
    crit = p.get("eligibilityModule", {}).get("eligibilityCriteria", "")
    return {"nct_id": nct_id, "title": p["identificationModule"].get("briefTitle", ""),
            "criteria": parse_criteria_text(crit)}


_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def number_note(text: str) -> tuple[str, bool]:
    """Return (numbered_note, was_auto_numbered).  Notes already using '<id>. ' lines are kept as they are
    (only if ids are contiguous from 0); otherwise the text is split into sentences and numbered from 0."""
    text = text.replace("\r", "").strip()
    sents = D.split_sentences(text)
    if sents and sorted(sents) == list(range(len(sents))) and re.match(r"^\s*0\.\s", text):
        return "\n".join(f"{i}. {sents[i]}" for i in sorted(sents)), False
    flat = re.sub(r"\s*\n\s*", " ", text)
    parts = [s.strip() for s in _SPLIT.split(flat) if s.strip()]
    return "\n".join(f"{i}. {s}" for i, s in enumerate(parts)), True


def overall_outcome(results: list[dict]) -> dict:
    """Rule-based aggregation of criterion-level labels (NOT a ground-truth label; see paper).

    1. any inclusion 'not included'  or exclusion 'excluded'          -> "Likely not eligible"
    2. else any 'not enough information' or invalid output            -> "Needs more information"
    3. else (all included / not excluded / not applicable)            -> "Potentially eligible"
    """
    bad = [r for r in results if r["label"] in ("not included", "excluded")]
    unk = [r for r in results if r["label"] in ("not enough information", "invalid")]
    ok = [r for r in results if r["label"] in ("included", "not excluded", "not applicable")]
    if bad:
        verdict, why = "Likely not eligible", f"{len(bad)} criterion/criteria failed (not included / excluded)."
    elif unk:
        verdict, why = "Needs more information", f"{len(unk)} criterion/criteria lack enough information."
    elif ok:
        verdict, why = "Potentially eligible", "No criterion failed and none lacks information (preliminary)."
    else:
        verdict, why = "No result", "No criteria were evaluated."
    return {"verdict": verdict, "reason": why, "n_failed": len(bad), "n_unknown": len(unk), "n_ok": len(ok),
            "rule": "fail if any inclusion 'not included' or exclusion 'excluded'; else unknown if any "
                    "'not enough information'/invalid; else potentially eligible. Preliminary, educational only."}
