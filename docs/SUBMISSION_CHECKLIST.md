# Rubric → evidence map and submission checklist

| Code | Evidence in this folder |
|---|---|
| A1 literature | `report/references.bib` (22 entries), `report/sections/02_related.tex`, `docs/RESEARCH_NOTES.md` |
| A2 model choice | paper §III-C, Table "candidates" (`scripts/make_candidate_table.py`, `results/metrics/candidate_table.json`), `docs/model_configs/` |
| A3 decisions/difficulties | `docs/DESIGN_DECISIONS.md` (alternatives, rejected options, Kaggle/CPU difficulties) |
| B1 dataset/split | `src/data.py`, `data/processed/split_manifest.json`, leakage asserts in `check_no_leakage` |
| B2 preprocessing | `src/data.py` (GPT-4 columns dropped in `load_raw`), `scripts/train.py:encode_example` |
| B3 fine-tuning | `scripts/train.py`, `configs/*.json`, `results/logs/train_*.csv`, `checkpoints/adapters/*/best` |
| B4 before/after | `results/metrics*/base_zeroshot_*.json` vs `finetuned_*.json` (val + test, same prompts/decoding) |
| C1–C2 metrics | `src/metrics.py`, `results/metrics*/finetuned_test.json`, paper Tables, confusion figs |
| C3 evidence | `src/metrics.py:evidence_report` (micro/macro, empty-gold policy) |
| C4 explanations/errors | `src/pipeline.py:explain`, `results/error_analysis_*_finetuned_test.md` (12 examples), paper §V-F |
| D1–D3 KV cache | `src/minillama.py` (KVCache, positions), `src/decode.py`, `tests/test_minillama.py`, `scripts/bench_kv.py`, `results/metrics*/bench_kv.json` |
| E1–E3 attention | `src/attention.py`, `tests/test_attention.py`, `scripts/bench_attention.py` |
| F1–F3 speculative | `src/decode.py:generate_speculative`, `scripts/bench_spec.py`, HF assisted-generation check |
| G1–G4 application | `app/`, `run_app.bat`, `tests/test_app.py`, report export, Compare tab |
| H1 paper | `report/main.tex` (+ sections/, generated/, figures/) |
| H2 repo/video | **you**: push to GitHub, host adapters, record video |
| H3 defense | **you**: be able to explain `minillama.py`, `attention.py`, `decode.py` live |

## You must do (cannot be automated)
- [ ] Put your name / roll number / university into `report/main.tex` (marked `[AUTHOR NAME]`).
- [ ] Compile `report/` on Overleaf (upload the whole folder; main file `main.tex`) and fix any warnings; confirm the PDF looks right.
- [ ] Verify every entry in `report/references.bib` against the publisher/arXiv page.
- [ ] Create the GitHub repo and push (`.gitignore` already excludes base weights and raw data).
- [ ] Upload adapters (`checkpoints/adapters/smollm2-1.7b-lora/best`, `smollm2-360m-gpu-lora/best`, `smollm2-360m-lora/best`, `smollm2-135m-lora/best`) to Hugging Face/Drive; put the link in `README.md` and the paper.
- [ ] Record the 5–7 minute demo (`docs/DEMO_SCRIPT.md`) and add the video link.
- [ ] Finish the author statement in `docs/AI_USE.md`.
- [ ] Submit on Google Classroom before 10 Nov 2026 (2 h grace period is not a target).
- [ ] Re-read `docs/DESIGN_DECISIONS.md` and the paper so you can defend every number.
