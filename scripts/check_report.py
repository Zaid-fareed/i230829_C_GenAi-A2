"""Static compile-readiness check for the LaTeX report (no TeX installation needed).

Checks: every \\input/\\includegraphics target exists, every \\ref has a \\label, every \\cite key is in references.bib,
every macro used from numbers files is defined, braces balance per file.
"""
import re
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "report"
tex = {p: p.read_text(encoding="utf-8") for p in R.rglob("*.tex")}
all_text = "\n".join(tex.values())
problems = []

# inputs and graphics
for p, s in tex.items():
    for m in re.finditer(r"\\input\{([^}]+)\}", s):
        t = m.group(1)
        if not ((R / t).exists() or (R / (t + ".tex")).exists()):
            problems.append(f"{p.name}: missing input {t}")
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", s):
        t = m.group(1)
        if not ((R / "figures" / t).exists() or (R / t).exists()):
            problems.append(f"{p.name}: missing figure {t}")

# labels / refs
labels = set(re.findall(r"\\label\{([^}]+)\}", all_text))
for p, s in tex.items():
    for m in re.finditer(r"\\ref\{([^}]+)\}", s):
        if m.group(1) not in labels:
            problems.append(f"{p.name}: \\ref to undefined label {m.group(1)}")
refs_used = set(re.findall(r"\\ref\{([^}]+)\}", all_text))
unused = sorted(l for l in labels if l not in refs_used)

# citations
bib = (R / "references.bib").read_text(encoding="utf-8")
keys = set(re.findall(r"@\w+\{([^,]+),", bib))
cited = set()
for m in re.finditer(r"\\cite\{([^}]+)\}", all_text):
    cited |= {k.strip() for k in m.group(1).split(",")}
for k in sorted(cited - keys):
    problems.append(f"cite key not in bib: {k}")
uncited = sorted(keys - cited)

# macros
defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", all_text))
builtin = {"BibTeX"}
used = set()
for p, s in tex.items():
    if p.parent.name == "generated":
        continue
    used |= set(re.findall(r"\\([A-Z][A-Za-z]+)\{\}", s))
for u in sorted(used - defined - builtin):
    problems.append(f"macro used but not defined: \\{u}{{}}")

# braces
for p, s in tex.items():
    body = re.sub(r"\\[{}]", "", s)
    if body.count("{") != body.count("}"):
        problems.append(f"{p.name}: unbalanced braces ({body.count('{')} vs {body.count('}')})")

print("files checked:", len(tex), "| labels:", len(labels), "| cite keys used:", len(cited), "of", len(keys))
print("labels never referenced (a figure/table not mentioned in text?):", unused)
print("bib entries never cited:", uncited)
print("PROBLEMS:" if problems else "no problems found")
for x in problems:
    print(" -", x)
sys.exit(1 if problems else 0)
