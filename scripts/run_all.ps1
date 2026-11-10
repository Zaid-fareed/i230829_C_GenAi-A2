# End-to-end reproduction (PowerShell). Adjust $py if needed.
#
# Configuration A (GPU, headline results): run on Kaggle/Colab or any CUDA machine with
#     kaggle/kaggle_gpu_run.ipynb   (or: python scripts/kaggle_pipeline.py)
# It trains SmolLM2-1.7B + 360M draft, evaluates, benchmarks, and writes results/metrics_gpu/.
# Copy results/metrics_gpu, results/logs and checkpoints/adapters back into this folder afterwards.
#
# Configuration B (laptop CPU, also the app's default model) is reproduced by this script.
# Expected wall-clock on a 4-core laptop: ~3.5 h (360M) + ~2.3 h (135M) training + ~1.5 h evaluation.
$ErrorActionPreference = "Stop"
$py = "python"
Set-Location (Split-Path $PSScriptRoot -Parent)

& $py scripts\download_data.py                  # 1. TrialGPT-Criterion-Annotations parquet (the only dataset used)
& $py src\data.py                               # 2. split manifest (seed 42, 37/8/8) + patient-leakage assertions
& $py scripts\download_models.py                # 3. open-weight base models
& $py tests\test_attention.py; & $py tests\test_minillama.py           # 4. correctness tests (attention, decoder, cache, speculative)
& $py scripts\train.py --config configs\finetune_360m.json   # 5. target fine-tuning (LoRA)
& $py scripts\train.py --config configs\finetune_135m.json   #    draft fine-tuning (LoRA)
& $py scripts\select_best.py                    # 6. copy best-validation epoch to .../best
& $py scripts\baselines.py                      # 7. majority-label baseline (train-split statistics only)
& $py scripts\evaluate.py --split validation --adapter none --tag base_zeroshot
& $py scripts\evaluate.py --split validation --adapter best --tag finetuned
& $py scripts\evaluate.py --split test --adapter none --tag base_zeroshot
& $py scripts\evaluate.py --split test --adapter best --tag finetuned  # 8. FINAL held-out evaluation (run once)

# 9. analysis, figures and LaTeX tables for both configurations (RUN selects results/metrics or results/metrics_gpu)
foreach ($run in "cpu", "gpu") {
    $env:RUN = $run
    & $py scripts\bootstrap_ci.py finetuned test; & $py scripts\bootstrap_ci.py base_zeroshot test
    & $py scripts\error_analysis.py finetuned test; & $py scripts\extra_stats.py
    & $py scripts\make_figures.py; & $py scripts\make_tables.py
}
& $py scripts\make_candidate_table.py           # 10. candidate-model table from config.json files
& $py scripts\check_report.py                   # 11. static check of the LaTeX sources (inputs, refs, cites)
& $py tests\test_app.py                         # 12. app API tests
& $py app\server.py                             # 13. start the web app at http://127.0.0.1:8000
