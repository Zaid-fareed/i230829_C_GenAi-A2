"""Drive the running web app in headless Edge (Chrome DevTools Protocol) and save screenshots for the paper.

Start the app first (python app/server.py --no-browser), then:  python scripts/take_screenshots.py
Writes report/figures/app_*.png (and results/figures/).
"""
import base64
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
PORT = 9333
URL = "http://127.0.0.1:8000"


class Page:
    def __init__(self):
        self.prof = tempfile.mkdtemp()
        self.proc = subprocess.Popen([EDGE, "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={self.prof}",
                                      "--window-size=1500,1100", "--hide-scrollbars", "--no-first-run", "about:blank"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
                ws = [t for t in tabs if t["type"] == "page"][0]["webSocketDebuggerUrl"]
                break
            except Exception:
                time.sleep(0.5)
        self.ws = websocket.create_connection(ws, max_size=None, timeout=300)
        self.i = 0

    def call(self, method, **params):
        self.i += 1
        self.ws.send(json.dumps({"id": self.i, "method": method, "params": params}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("id") == self.i:
                return m.get("result", {})

    def js(self, expr):
        r = self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        return r.get("result", {}).get("value")

    def wait(self, cond, timeout=600, step=1.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.js(cond):
                return True
            time.sleep(step)
        raise TimeoutError(cond)

    def shot(self, name, full=True):
        h = self.js("Math.max(document.body.scrollHeight, 900)") if full else 1100
        self.call("Emulation.setDeviceMetricsOverride", width=1500, height=int(min(h, 3200)), deviceScaleFactor=1, mobile=False)
        data = self.call("Page.captureScreenshot", format="png")["data"]
        for d in (OUT, ROOT / "results" / "figures"):
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{name}.png").write_bytes(base64.b64decode(data))
        print("saved", name)

    def close(self):
        try:
            self.ws.close()
        finally:
            self.proc.terminate()


def main():
    p = Page()
    try:
        p.call("Page.enable")
        p.call("Page.navigate", url=URL)
        time.sleep(2)
        p.wait("document.querySelector('#status').textContent.includes('ready')", timeout=300)
        p.wait("document.querySelector('#sample').options.length > 1")
        # ---- Screen tab: sample patient, sample trial, 6 criteria (keeps the run short)
        p.js("(()=>{const s=document.querySelector('#sample'); s.value='0'; s.onchange({target:s});"
             "const t=document.querySelector('#trialSel'); t.value='0'; t.onchange({target:t}); CRITS=CRITS.slice(0,6); renderCrits();})()")
        p.shot("app_01_inputs", full=False)
        p.js("document.querySelector('#btnRun').click()")
        p.wait("document.querySelector('#runMsg').textContent.startsWith('Done')", timeout=900)
        p.js("document.querySelectorAll('#results .card')[1].click()")
        time.sleep(1)
        p.shot("app_02_results")
        # ---- validation error recovery
        p.js("document.querySelector('#note').value='too short'; document.querySelector('#btnRun').click()")
        time.sleep(1)
        p.shot("app_03_error_recovery", full=False)
        # ---- Compare tab
        p.js("(()=>{const s=document.querySelector('#sample'); s.value='0'; s.onchange({target:s}); document.querySelector('#note').value=SAMPLES[0].note;})()")
        p.js("document.querySelector('#nav button[data-t=compare]').click()")
        time.sleep(1)
        p.js("document.querySelector('#bRep').value=1; document.querySelector('#btnBench').click()")
        p.wait("document.querySelector('#benchOut').innerText.length > 50 || document.querySelector('#benchMsg').innerText.includes('Enter')", timeout=1500)
        p.shot("app_04_compare")
        # ---- Experiments tab
        p.js("document.querySelector('#nav button[data-t=exp]').click()")
        time.sleep(3)
        p.shot("app_05_experiments")
        # ---- exported report
        p.js("document.querySelector('#nav button[data-t=screen]').click()")
    finally:
        p.close()


if __name__ == "__main__":
    main()
