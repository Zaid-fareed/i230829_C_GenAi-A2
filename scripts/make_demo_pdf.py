"""Build a phone-sized PDF of docs/DEMO_SCRIPT.md (narrow pages, large text) using Markdown -> HTML -> headless Edge.

Output: docs/DEMO_SCRIPT_mobile.pdf
"""
import re
import subprocess
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "DEMO_SCRIPT.md"
OUT = ROOT / "docs" / "DEMO_SCRIPT_mobile.pdf"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

html_body = markdown.markdown(SRC.read_text(encoding="utf-8"), extensions=["fenced_code", "sane_lists"])

# colour the three labels so they stand out on a small screen
html_body = re.sub(r"<strong>DO</strong>", '<span class="tag do">DO</span>', html_body)
html_body = re.sub(r"<strong>POINT</strong>", '<span class="tag point">POINT</span>', html_body)
html_body = re.sub(r"<strong>SAY</strong>", '<span class="tag say">SAY</span>', html_body)

css = """
@page { size: 100mm 200mm; margin: 6mm 5mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 11.5pt; line-height: 1.38; color: #111; }
h1 { font-size: 16pt; margin: 0 0 4mm; color: #0b3d91; }
h2 { font-size: 13.5pt; background: #0b3d91; color: #fff; padding: 2mm 3mm; border-radius: 2mm; margin: 6mm 0 3mm; page-break-after: avoid; }
p, li { margin: 0 0 2.2mm; }
ul, ol { padding-left: 5mm; }
hr { border: 0; border-top: 1px solid #bbb; margin: 4mm 0; }
code { font-family: Consolas, monospace; background: #eef1f6; padding: 0 1mm; border-radius: 1mm; font-size: 10.2pt; word-break: break-word; }
pre { background: #f3f5f9; border-left: 1mm solid #0b3d91; padding: 2mm 3mm; margin: 2mm 0 3mm; white-space: pre-wrap; word-break: break-word; font-size: 9.8pt; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 9.8pt; }
.tag { display: inline-block; font-weight: 700; font-size: 9.5pt; color: #fff; padding: 0.2mm 2mm; border-radius: 1.5mm; margin-right: 1mm; }
.do { background: #1b7f3b; } .point { background: #c25b00; } .say { background: #6a1b9a; }
"""
doc = f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><style>{css}</style></head><body>{html_body}</body></html>"

with tempfile.TemporaryDirectory() as d:
    f = Path(d) / "script.html"
    f.write_text(doc, encoding="utf-8")
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={OUT}", f.as_uri()],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
print("wrote", OUT, OUT.stat().st_size // 1024, "KB")
