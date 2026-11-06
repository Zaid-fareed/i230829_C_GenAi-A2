"""API smoke tests for the web app (uses FastAPI's in-process client; loads the real models).

Run: python tests/test_app.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "app"))

from fastapi.testclient import TestClient  # noqa: E402
import server  # noqa: E402
import trials as T  # noqa: E402


def test_parsing_helpers():
    items = T.parse_criteria_text("Inclusion Criteria:\n* Age 18-75\n* Confirmed asthma\n  for at least 1 year\nExclusion Criteria:\n- Pregnancy\n- Current smoker")
    assert [i["type"] for i in items] == ["inclusion", "inclusion", "exclusion", "exclusion"]
    assert items[1]["text"] == "Confirmed asthma for at least 1 year"
    numbered, auto = T.number_note("A man has cough. He smokes daily. No fever.")
    assert auto and numbered.splitlines()[2].startswith("2. ")
    numbered2, auto2 = T.number_note("0. First.\n1. Second.")
    assert not auto2 and numbered2 == "0. First.\n1. Second."
    assert T.overall_outcome([{"label": "included"}, {"label": "excluded"}])["verdict"] == "Likely not eligible"
    assert T.overall_outcome([{"label": "included"}, {"label": "not enough information"}])["verdict"] == "Needs more information"
    assert T.overall_outcome([{"label": "included"}, {"label": "not excluded"}])["verdict"] == "Potentially eligible"


def test_api():
    with TestClient(server.app) as c:
        for _ in range(120):
            if c.get("/api/status").json()["status"] in ("ready", "error"):
                break
            time.sleep(1)
        st = c.get("/api/status").json()
        assert st["status"] == "ready", st
        assert c.get("/").status_code == 200
        # input validation / recovery from invalid input
        assert c.post("/api/predict", json={"note": "short", "criterion": {"type": "inclusion", "text": "x"}}).status_code == 422
        assert c.post("/api/predict", json={"note": "A" * 40, "criterion": {"type": "weird", "text": "abc"}}).status_code == 422
        assert c.post("/api/parse_criteria", json={"text": "   "}).status_code == 422
        assert c.post("/api/fetch_trial", json={"nct_id": "nope"}).status_code == 422
        s = c.get("/api/samples").json()["samples"]
        assert len(s) == 8
        note = s[0]["note"]
        crit = s[0]["trials"][0]["criteria"][0]
        outs = {}
        for mode in ("uncached", "kv_cache", "kv_cache_tiled", "speculative"):
            r = c.post("/api/predict", json={"note": note, "criterion": {"type": crit["type"], "text": crit["text"]}, "mode": mode})
            assert r.status_code == 200, r.text
            outs[mode] = r.json()["raw_output"]
            print(mode, "->", outs[mode], f"({r.json()['latency_s']:.2f}s)")
        assert len(set(outs.values())) == 1, outs
        res = [c.post("/api/predict", json={"note": note, "criterion": {"type": crit["type"], "text": crit["text"]}}).json()]
        assert c.post("/api/overall", json={"results": res}).status_code == 200
        for fmt in ("html", "md", "json"):
            r = c.post("/api/report", json={"note": note, "results": res, "mode": "kv_cache", "format": fmt, "trial_label": "t"})
            assert r.status_code == 200 and "NOT a clinical enrollment decision" in r.text
        assert c.post("/api/report", json={"note": note, "results": [], "mode": "kv_cache"}).status_code == 422
        b = c.post("/api/benchmark", json={"note": note, "criterion": {"type": crit["type"], "text": crit["text"]}, "repeats": 1, "modes": ["kv_cache", "speculative"]})
        assert b.status_code == 200 and all(r["tokens_match_first_mode"] for r in b.json()["rows"]), b.text


if __name__ == "__main__":
    test_parsing_helpers(); print("parsing ok")
    test_api(); print("api ok")
