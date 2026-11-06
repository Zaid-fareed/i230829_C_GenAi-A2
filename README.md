# Criterion-level clinical-trial eligibility screening with a small Transformer

Generative AI – Assignment 2. A locally executable system that, given a patient note and a trial criterion,
predicts the expert eligibility label **and** the supporting note-sentence IDs, built on the
[TrialGPT Criterion Annotations](https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations) dataset only.

> **Educational screening prototype – not a clinical enrollment decision.**

| Part | Where |
|---|---|
| Fine-tuned model (SmolLM2-360M-Instruct + LoRA) and 135M draft | `scripts/train.py`, `configs/` |
| Own Llama decoder, explicit KV cache, LoRA merge | `src/minillama.py` |
| Uncached / cached / speculative decoding loops | `src/decode.py` |
| Reference + tiled online-softmax attention | `src/attention.py` |
| Data, fixed split, prompt/target format | `src/data.py` → `data/processed/split_manifest.json` |
| Metrics (classification + evidence) | `src/metrics.py` |
| Browser app | `app/` (`run_app.bat`) |
| Tests | `tests/` |
| Benchmarks / evaluation | `scripts/bench_*.py`, `scripts/evaluate.py` → `results/` |
| IEEE LaTeX paper | `report/main.tex` |
| Research notes, design decisions, AI-use, demo script, checklist | `docs/` |

## Results at a glance (held-out test split: 8 patients, 206 criteria)
| Configuration | Accuracy | Macro-F1 | Evidence micro-F1 |
|---|---|---|---|
| Majority label per criterion type (baseline) | 0.626 | 0.272 | – |
| **A (GPU)** SmolLM2-1.7B zero-shot | 0.146 | 0.111 | 0.151 |
| **A (GPU)** SmolLM2-1.7B + LoRA | **0.709** | **0.460** | **0.384** |
| **B (CPU)** SmolLM2-360M zero-shot | 0.437 | 0.234 | 0.000 |
| **B (CPU)** SmolLM2-360M + LoRA | 0.626 | 0.333 | 0.123 |

Honest reading: the accuracy gain over the majority baseline is not statistically established (patient-level bootstrap
intervals overlap), the gain is confined to *inclusion* criteria, and on *exclusion* criteria the models answer
"not excluded" almost always. Details, per-class tables, confusion matrices and error analysis are in the paper.
KV cache: exact (token-identical) and up to ~19× faster than recomputation at 512+128 tokens (GPU);
tiled attention: matches the reference to ~1e-6, memory grows linearly, slower than fused kernels in pure PyTorch;
speculative decoding: exact, but slower than greedy for these draft/target pairs (explained in the paper).

## Folder layout
```
src/        library code (data, metrics, attention, minillama, decode, pipeline, trials, memprobe)
scripts/    download, train, evaluate, benchmarks, figures, tables, error analysis
configs/    fine-tuning configs (360M target, 135M draft)
app/        FastAPI server + static single-page UI
tests/      attention, decoder/cache/speculative, app API tests
data/       raw parquet (downloaded) and processed split manifest
checkpoints/ base models (downloaded), LoRA adapters (trained)
results/    logs/ (training curves, run output), metrics/ (JSON + predictions), figures/
report/     LaTeX source, bibliography, generated tables/figures
docs/       notes, decisions, AI-use appendix, demo script, submission checklist
```

## Setup (Windows, Python 3.10+; no GPU needed)
```powershell
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/download_data.py          # dataset parquet -> data/raw
python src/data.py                       # writes data/processed/split_manifest.json (seed 42, 37/8/8) + leakage asserts
python scripts/download_models.py        # SmolLM2-360M/135M-Instruct (+ Qwen2.5-0.5B for the comparison)
```

## Reproduce training and evaluation
```powershell
python scripts/train.py --config configs/finetune_360m.json   # ~3 h on 4 CPU cores
python scripts/train.py --config configs/finetune_135m.json   # draft, ~1 h
python scripts/select_best.py                                 # copies best-validation epoch to .../best
python scripts/evaluate.py --split validation --adapter none --tag base_zeroshot
python scripts/evaluate.py --split validation --adapter best --tag finetuned
python scripts/evaluate.py --split test --adapter none --tag base_zeroshot   # final: run once
python scripts/evaluate.py --split test --adapter best --tag finetuned
python scripts/bootstrap_ci.py finetuned test ; python scripts/error_analysis.py finetuned test
python scripts/baselines.py ; python scripts/candidate_compare.py
python scripts/bench_kv.py ; python scripts/bench_attention.py ; python scripts/bench_spec.py
python scripts/make_figures.py ; python scripts/make_tables.py
```
`scripts/run_all.ps1` chains everything. The test split is read only by `evaluate.py --split test` and the benchmark scripts (speed only).

## Run the application
```powershell
python app/server.py          # or double-click run_app.bat
# open http://127.0.0.1:8000  (models load in ~15 s; the status pill turns green)
```
Tabs: **Screen a patient** (note → criteria → per-criterion label/evidence/explanation → export HTML/Markdown/JSON),
**Compare decoding** (live uncached vs KV-cache vs tiled attention vs speculative), **Experiments** (stored results),
**Help**. If the adapter is missing the app falls back to the base model and says so in the header.

## Model weights
`checkpoints/adapters/smollm2-360m-lora/best/` and `.../smollm2-135m-lora/best/` hold the LoRA adapters (~35 MB / ~13 MB).
Download link for hosting outside Git: **TODO – add Hugging Face / Drive URL before submission** (see `docs/SUBMISSION_CHECKLIST.md`).
Base weights are fetched from the Hugging Face Hub by `scripts/download_models.py`.

## Tests
```powershell
python tests/test_attention.py ; python tests/test_minillama.py ; python tests/test_app.py
```

## Data hygiene
GPT-4 columns (`gpt4_explanation`, `gpt4_sentences`, `gpt4_eligibility`, `explanation_correctness`) are removed in
`src/data.py:load_raw` before anything else touches the data. Annotation 883 has no criterion text and is excluded from
training/evaluation (recorded in the manifest).
