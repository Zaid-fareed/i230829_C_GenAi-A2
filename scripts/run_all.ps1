# End-to-end reproduction on a CPU-only Windows machine (PowerShell).  Adjust $py if needed.
# Expected wall-clock on a 4-core laptop: ~3.5 h fine-tuning (360M) + ~1.2 h (135M) + ~1 h evaluation/benchmarks.
$ErrorActionPreference = "Stop"
$py = "python"
Set-Location (Split-Path $PSScriptRoot -Parent)

& $py scripts\download_data.py                  # 1. TrialGPT-Criterion-Annotations parquet
& $py src\data.py                               # 2. split manifest (seed 42, 37/8/8) + leakage checks
& $py scripts\download_models.py                # 3. open-weight base models
& $py scripts\train.py --config configs\finetune_360m.json   # 4. target fine-tuning (LoRA)
& $py scripts\train.py --config configs\finetune_135m.json   #    draft fine-tuning (LoRA)
& $py scripts\select_best.py                    # 5. copy best-validation epoch to .../best
& $py tests\test_attention.py; & $py tests\test_minillama.py           # 6. correctness tests
& $py scripts\evaluate.py --split validation --adapter none --tag base_zeroshot
& $py scripts\evaluate.py --split validation --adapter best --tag finetuned
& $py scripts\evaluate.py --split test --adapter none --tag base_zeroshot
& $py scripts\evaluate.py --split test --adapter best --tag finetuned  # 7. FINAL held-out evaluation (run once)
& $py scripts\bench_kv.py; & $py scripts\bench_attention.py; & $py scripts\bench_spec.py   # 8. benchmarks
& $py scripts\make_figures.py                   # 9. figures + tables for the paper
& $py app\server.py                             # 10. start the web app at http://127.0.0.1:8000
