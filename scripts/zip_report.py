"""Zip the report/ folder (LaTeX sources, generated tables, figures, bibliography) for upload to Overleaf.

Output: ../2_overleaf_upload/report.zip  (files at the zip root, so Overleaf finds main.tex directly).
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
OUT = ROOT.parent / "2_overleaf_upload" / "report.zip"
OUT.parent.mkdir(exist_ok=True)
SKIP_SUFFIX = {".aux", ".log", ".out", ".bbl", ".blg", ".synctex.gz"}
n = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(REPORT.rglob("*")):
        if p.is_file() and p.suffix not in SKIP_SUFFIX:
            z.write(p, p.relative_to(REPORT).as_posix())
            n += 1
with zipfile.ZipFile(OUT) as z:
    names = z.namelist()
print(n, "files,", round(OUT.stat().st_size / 1e6, 2), "MB ->", OUT)
print("main.tex at root:", "main.tex" in names)
print("top-level folders:", sorted({x.split('/')[0] for x in names if '/' in x}))
print("sections:", sorted(x for x in names if x.startswith("sections/")))
