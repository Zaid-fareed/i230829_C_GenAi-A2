"""Copy the best validation epoch of an adapter run to <run>/best (selection already recorded in best.json)."""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for run in sys.argv[1:] or ["checkpoints/adapters/smollm2-360m-lora", "checkpoints/adapters/smollm2-135m-lora"]:
    d = ROOT / run
    if not (d / "best.json").exists():
        print("skip (not finished):", d); continue
    ep = json.loads((d / "best.json").read_text())["best_epoch"]
    dst = d / "best"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(d / f"epoch{ep}", dst)
    print(f"{d.name}: best = epoch {ep} -> {dst}")
