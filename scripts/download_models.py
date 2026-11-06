"""Download the open-weight base models used in the project into checkpoints/base/."""
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    "HuggingFaceTB/SmolLM2-360M-Instruct",   # target
    "HuggingFaceTB/SmolLM2-135M-Instruct",   # draft (same tokenizer)
    "Qwen/Qwen2.5-0.5B-Instruct",            # alternative candidate (zero-shot comparison only)
]

if __name__ == "__main__":
    names = sys.argv[1:] or MODELS
    for m in names:
        dest = ROOT / "checkpoints" / "base" / m.split("/")[1]
        snapshot_download(m, local_dir=dest, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.md"])
        print("done", m, dest, flush=True)
