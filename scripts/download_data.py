"""Download the (only permitted) TrialGPT Criterion Annotations parquet file to data/raw/."""
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
p = hf_hub_download("ncbi/TrialGPT-Criterion-Annotations", "data/train-00000-of-00001.parquet",
                    repo_type="dataset", local_dir=ROOT / "data" / "raw")
print("saved", p)
