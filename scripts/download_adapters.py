"""Download the trained LoRA adapters from the Hugging Face Hub into checkpoints/adapters/<name>/best/.

usage:  python scripts/download_adapters.py <hf-user>/<repo-name>
The base models are downloaded separately with scripts/download_models.py.
"""
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ["smollm2-1.7b-lora", "smollm2-360m-gpu-lora", "smollm2-360m-lora", "smollm2-135m-lora"]


def main():
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        raise SystemExit(__doc__)
    repo = sys.argv[1]
    tmp = Path(snapshot_download(repo, repo_type="model"))
    for name in ADAPTERS:
        src = tmp / name
        if not (src / "adapter_model.safetensors").exists():
            print("missing in repo:", name)
            continue
        dst = ROOT / "checkpoints" / "adapters" / name / "best"
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("adapter_config.json", "adapter_model.safetensors"):
            shutil.copy2(src / f, dst / f)
        print("installed", name, "->", dst)


if __name__ == "__main__":
    main()
