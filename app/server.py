"""Browser application: FastAPI backend + static single-page UI.

Start:  python app/server.py        (then open http://127.0.0.1:8000)
"""
from __future__ import annotations

import datetime as dt
import hashlib
import html
import json
import statistics as st
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field, field_validator  # noqa: E402

import data as D  # noqa: E402
import trials as T  # noqa: E402

DISCLAIMER = ("Educational screening prototype - NOT a clinical enrollment decision. Outputs come from a small "
              "fine-tuned language model, can be wrong, and must never be used to make decisions about care or trial entry.")

app = FastAPI(title="Trial Eligibility Screening Prototype")
STATE = {"engine": None, "status": "idle", "error": None}
LOCK = threading.Lock()


def _load_engine():
    try:
        STATE["status"] = "loading"
        from pipeline import Engine
        STATE["engine"] = Engine()
        STATE["status"] = "ready"
    except Exception as e:  # surfaced to the UI
        STATE["status"] = "error"
        STATE["error"] = f"{type(e).__name__}: {e}"


@app.on_event("startup")
def _startup():
    threading.Thread(target=_load_engine, daemon=True).start()


def engine():
    if STATE["status"] != "ready":
        raise HTTPException(503, f"Model not ready (status: {STATE['status']}). {STATE['error'] or 'Please wait a few seconds and retry.'}")
    return STATE["engine"]


# ---------------------------------------------------------------- schemas / validation
class Criterion(BaseModel):
    type: str
    text: str = Field(min_length=3, max_length=2000)

    @field_validator("type")
    @classmethod
    def _t(cls, v):
        v = v.strip().lower()
        if v not in ("inclusion", "exclusion"):
            raise ValueError("criterion type must be 'inclusion' or 'exclusion'")
        return v


class PredictReq(BaseModel):
    note: str = Field(min_length=20, max_length=8000)
    criterion: Criterion
    mode: str = "kv_cache"


class BenchReq(BaseModel):
    note: str = Field(min_length=20, max_length=8000)
    criterion: Criterion
    modes: list[str] = ["uncached", "kv_cache", "kv_cache_tiled", "speculative"]
    repeats: int = Field(default=2, ge=1, le=5)
    k: int = Field(default=4, ge=1, le=12)
    forced_new_tokens: int = Field(default=0, ge=0, le=128)


class ReportReq(BaseModel):
    note: str
    trial_label: str = ""
    results: list[dict]
    mode: str
    format: str = "html"


# ---------------------------------------------------------------- endpoints
@app.get("/api/status")
def status():
    e = STATE["engine"]
    from pipeline import MODES, MAX_NEW
    return {"status": STATE["status"], "error": STATE["error"], "disclaimer": DISCLAIMER, "modes": MODES,
            "adapter": e.adapter_name if e else None, "max_new_tokens": MAX_NEW,
            "model": "SmolLM2-360M-Instruct + LoRA (target), SmolLM2-135M-Instruct + LoRA (draft)"}


@app.post("/api/number_note")
def number_note(body: dict):
    text = str(body.get("note", ""))
    if len(text.strip()) < 20:
        raise HTTPException(422, "The patient note is too short (minimum 20 characters).")
    numbered, auto = T.number_note(text)
    return {"numbered": numbered, "auto_numbered": auto, "n_sentences": numbered.count("\n") + 1}


@app.post("/api/parse_criteria")
def parse_criteria(body: dict):
    items = T.parse_criteria_text(str(body.get("text", "")))
    if not items:
        raise HTTPException(422, "No criteria could be parsed. Use 'Inclusion Criteria:' / 'Exclusion Criteria:' headings and one bullet per criterion.")
    return {"criteria": items}


@app.post("/api/fetch_trial")
def fetch_trial(body: dict):
    try:
        return T.fetch_clinicaltrials(str(body.get("nct_id", "")))
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(502, f"Could not reach ClinicalTrials.gov ({type(e).__name__}). Paste the criteria manually instead.")


_SAMPLES = None


@app.get("/api/samples")
def samples():
    """Held-out TEST patients with their trials (for demos).  Expert labels are shown only for comparison."""
    global _SAMPLES
    if _SAMPLES is None:
        df = D.load_split("test")
        out = []
        for pid, g in df.groupby("patient_id"):
            note = g.iloc[0]["note"]
            trials = []
            for tid, gt in g.groupby("trial_id"):
                trials.append({"trial_id": tid, "title": gt.iloc[0]["trial_title"],
                               "criteria": [{"type": r["criterion_type"], "text": r["criterion_text"],
                                             "expert_label": r["expert_eligibility"],
                                             "expert_evidence": r["expert_sentences"]} for _, r in gt.iterrows()]})
            out.append({"patient_id": pid, "note": note, "trials": trials})
        _SAMPLES = out
    return {"samples": _SAMPLES}


@app.post("/api/predict")
def predict(req: PredictReq):
    eng = engine()
    numbered, _ = T.number_note(req.note)
    from pipeline import MODES
    if req.mode not in MODES:
        raise HTTPException(422, f"unknown mode {req.mode}")
    try:
        with LOCK:
            res = eng.predict(numbered, req.criterion.type, req.criterion.text, mode=req.mode)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return res


@app.post("/api/overall")
def overall(body: dict):
    return T.overall_outcome(body.get("results", []))


@app.post("/api/benchmark")
def benchmark(req: BenchReq):
    """Run the same prompt through several decoding modes and compare tokens, latency and speed."""
    eng = engine()
    from memprobe import PeakRSS
    from pipeline import MODES
    numbered, _ = T.number_note(req.note)
    ids = eng.encode(numbered, req.criterion.type, req.criterion.text)
    max_new = max(req.forced_new_tokens, 48) if req.forced_new_tokens else 48
    min_new = req.forced_new_tokens
    rows, ref_tokens = [], None
    with LOCK:
        for mode in req.modes:
            if mode not in MODES:
                raise HTTPException(422, f"unknown mode {mode}")
            eng.generate(ids[:24], mode, max_new=2, min_new=2)               # warm-up
            runs = []
            with PeakRSS() as pk:
                for _ in range(req.repeats):
                    runs.append(eng.generate(ids, mode, max_new, min_new, k=req.k))
            r = runs[0]
            if ref_tokens is None:
                ref_tokens = r.tokens
            row = {"mode": mode, "n_prompt_tokens": len(ids), "n_new_tokens": len(r.tokens),
                   "latency_s": st.median(x.total_s for x in runs), "prefill_s": st.median(x.prefill_s for x in runs),
                   "decode_s": st.median(x.decode_s for x in runs), "tokens_per_s": st.median(x.tok_per_s for x in runs),
                   "peak_delta_mb": pk.delta_mb, "tokens_match_first_mode": r.tokens == ref_tokens,
                   "target_forward_passes": r.n_forward_target, "draft_forward_passes": r.n_forward_draft,
                   "output": eng.tok.decode([t for t in r.tokens if t not in eng.eos])}
            if mode == "speculative":
                row.update({"acceptance_rate": r.stats["acceptance_rate"], "k": r.stats["k"],
                            "mean_accepted_per_round": r.stats["mean_accepted_per_round"]})
            rows.append(row)
    base = rows[0]["latency_s"]
    for row in rows:
        row["speedup_vs_first"] = base / row["latency_s"] if row["latency_s"] else None
    return {"rows": rows, "note": "Timings are medians on this machine's CPU (fp32). Greedy outputs must match across modes."}


@app.get("/api/stored_benchmarks")
def stored():
    files = {"bench_kv": "bench_kv.json", "bench_attention": "bench_attention.json", "bench_spec": "bench_spec.json",
             "finetuned_test": "finetuned_test.json", "base_zeroshot_test": "base_zeroshot_test.json"}
    out = {}
    for key, fname in files.items():
        p = ROOT / "results" / "metrics" / fname
        if p.exists():
            out[key] = json.loads(p.read_text())
    return out


def _provenance(mode):
    e = STATE["engine"]
    ad = Path(__file__).resolve().parents[1] / "checkpoints" / "adapters" / "smollm2-360m-lora" / "best" / "adapter_model.safetensors"
    sha = hashlib.sha256(ad.read_bytes()).hexdigest()[:16] if ad.exists() else "n/a"
    return {"generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "model": "SmolLM2-360M-Instruct + LoRA adapter", "adapter_sha256_prefix": sha,
            "decoding": f"greedy, max_new_tokens=48, mode={mode}, fp32 CPU", "adapter": e.adapter_name if e else None}


@app.post("/api/report")
def report(req: ReportReq):
    if not req.results:
        raise HTTPException(422, "No results to export - run a screening first.")
    prov = _provenance(req.mode)
    ov = T.overall_outcome(req.results)
    note_hash = hashlib.sha256(req.note.encode()).hexdigest()[:12]
    if req.format == "json":
        return JSONResponse({"disclaimer": DISCLAIMER, "provenance": prov, "note_sha256_prefix": note_hash,
                             "trial": req.trial_label, "overall": ov, "patient_note": req.note, "criteria": req.results})
    if req.format == "md":
        lines = [f"# Eligibility screening report", f"> {DISCLAIMER}", "", f"- Trial: {req.trial_label or 'n/a'}",
                 f"- Generated (UTC): {prov['generated_utc']}", f"- Model: {prov['model']} (adapter sha256 {prov['adapter_sha256_prefix']})",
                 f"- Decoding: {prov['decoding']}", f"- Patient note hash: {note_hash}", "",
                 f"## Preliminary outcome: {ov['verdict']}", ov["reason"], f"_Rule: {ov['rule']}_", "", "## Criteria"]
        for i, r in enumerate(req.results, 1):
            lines += [f"### {i}. [{r['criterion_type']}] {r['criterion_text']}", f"- **Label:** {r['label']}",
                      "- **Evidence:** " + ("; ".join(f"[{e['id']}] {e['text']}" for e in r.get('evidence_text', [])) or "none cited"),
                      f"- **Explanation:** {r['explanation']}", ""]
        lines += ["## Patient note", "```", req.note, "```"]
        return PlainTextResponse("\n".join(lines))
    esc = html.escape
    badge = {"Potentially eligible": "#1b7f3b", "Likely not eligible": "#b3261e", "Needs more information": "#9a6700"}.get(ov["verdict"], "#444")
    rows = ""
    for i, r in enumerate(req.results, 1):
        ev = "".join(f"<li><b>[{e['id']}]</b> {esc(e['text'])}</li>" for e in r.get("evidence_text", [])) or "<li><i>no sentence cited</i></li>"
        rows += (f"<section><h3>{i}. <small>{esc(r['criterion_type'])}</small> {esc(r['criterion_text'])}</h3>"
                 f"<p><b>Predicted label:</b> <span class='lab'>{esc(r['label'])}</span></p><p><b>Evidence</b></p><ul>{ev}</ul>"
                 f"<p><b>Explanation:</b> {esc(r['explanation'])}</p></section>")
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Eligibility screening report</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:860px;margin:30px auto;padding:0 16px;color:#1d1d1f}}
.warn{{background:#fff4ce;border:1px solid #e0b400;padding:10px 14px;border-radius:6px}}
section{{border:1px solid #ddd;border-radius:8px;padding:6px 16px;margin:12px 0;page-break-inside:avoid}}
.lab{{background:#eef;padding:2px 8px;border-radius:10px}} .verdict{{color:#fff;background:{badge};padding:8px 14px;border-radius:8px;display:inline-block}}
pre{{white-space:pre-wrap;background:#f6f6f6;padding:10px;border-radius:6px}} small{{color:#666;text-transform:uppercase}}</style></head><body>
<h1>Eligibility screening report</h1><p class="warn"><b>{esc(DISCLAIMER)}</b></p>
<p><b>Trial:</b> {esc(req.trial_label or 'n/a')}<br><b>Generated (UTC):</b> {prov['generated_utc']}<br>
<b>Model:</b> {esc(prov['model'])} (adapter sha256 prefix {prov['adapter_sha256_prefix']})<br><b>Decoding:</b> {esc(prov['decoding'])}<br>
<b>Patient note hash:</b> {note_hash}</p>
<h2>Preliminary outcome</h2><p class="verdict">{esc(ov['verdict'])}</p><p>{esc(ov['reason'])}<br><small style="text-transform:none">Rule: {esc(ov['rule'])}</small></p>
<h2>Criterion-level results</h2>{rows}<h2>Patient note (numbered)</h2><pre>{esc(req.note)}</pre></body></html>"""
    return HTMLResponse(doc)


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


if __name__ == "__main__":
    import argparse
    import webbrowser

    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    if not a.no_browser:
        threading.Timer(2.0, lambda: webbrowser.open(f"http://{a.host}:{a.port}")).start()
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")
