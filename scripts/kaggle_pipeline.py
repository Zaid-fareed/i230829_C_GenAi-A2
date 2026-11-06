"""One-shot GPU pipeline for Kaggle/Colab (also works on any CUDA machine).

Trains the 1.7B target and the 360M draft, evaluates before/after on validation+test, runs all benchmarks on the GPU,
and writes everything under results/metrics_gpu/ and checkpoints/adapters/.  Run from the project root:
    python scripts/kaggle_pipeline.py [--skip-train]
"""
import argparse, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
env = dict(os.environ, TRIAL_PROFILE="configs/profile_gpu.json", PYTHONPATH=str(ROOT / "src"))
py = sys.executable


def run(*a):
    print(">>", " ".join(map(str, a)), flush=True)
    subprocess.run([py, *map(str, a)], check=True, env=env)


ap = argparse.ArgumentParser()
ap.add_argument("--skip-train", action="store_true")
args = ap.parse_args()
(ROOT / "results" / "metrics_gpu").mkdir(parents=True, exist_ok=True)
run("scripts/download_data.py")
run("src/data.py")
run("scripts/download_models.py", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct")
if not args.skip_train:
    run("scripts/train.py", "--config", "configs/finetune_1.7b_gpu.json")
    run("scripts/train.py", "--config", "configs/finetune_360m_gpu.json")
    run("scripts/select_best.py", "checkpoints/adapters/smollm2-1.7b-lora", "checkpoints/adapters/smollm2-360m-gpu-lora")
for split in ("validation", "test"):
    run("scripts/evaluate.py", "--split", split, "--adapter", "none", "--tag", "base_zeroshot")
    run("scripts/evaluate.py", "--split", split, "--adapter", "best", "--tag", "finetuned")
run("scripts/bench_kv.py", "--out", "results/metrics_gpu/bench_kv.json", "--threads", "2")
run("scripts/bench_attention.py", "--out", "results/metrics_gpu/bench_attention.json", "--lengths", "128", "256", "512", "1024", "2048", "4096")
run("scripts/bench_spec.py", "--out", "results/metrics_gpu/bench_spec.json", "--n", "32", "--threads", "2")
print("ALL DONE")
