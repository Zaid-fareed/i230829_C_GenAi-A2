# AI-use appendix

This project was carried out by the author with an AI coding and writing assistant: Claude (Anthropic), used
through the Claude Code assistant. The assistant was used for drafting and debugging; the direction, decisions,
running, testing and checking were done by the author.

## The author's work
* **Direction and requirements.** Worked from the assignment brief, set the goals for each part (data split,
  fine-tuning, KV cache, tiled attention, speculative decoding, web app, report) and gave the input for each step.
* **Design decisions.** Decided to move the main training and benchmarks to a free Kaggle GPU for a larger model
  (SmolLM2-1.7B), checked whether the chosen model family was allowed and compared it with Qwen2.5, and decided which
  experiments were needed (for example, skipping the CPU benchmark grid because the GPU run already covers it).
* **Running the GPU experiments.** Set up and ran the Kaggle notebook (dataset upload, GPU and internet settings),
  ran fine-tuning, evaluation and benchmarks, and fixed the problems met there (dataset path, `torchao` version
  conflict, CUDA out-of-memory) together with the assistant, then brought the results back into the project.
* **Testing and verification.** Ran and tested the browser application with sample and new patient cases, checked
  its answers against the expert labels and the error handling, and reviewed the reported results.
* **Report.** Reviewed and revised the report, checked all 22 references against their sources (leading to fixes),
  supplied the author details and compiled the paper on Overleaf.
* **Release.** Published the trained adapters on Hugging Face, reviewed the repository and the commit history, and
  recorded the demonstration video.

## The assistant's part
* Drafted the source code (data pipeline, metrics, decoder and KV cache, attention, speculative decoding,
  benchmarks, evaluation scripts, web application), the LaTeX report and the documentation, from the author's direction.
* Helped diagnose and fix errors, including those met on Kaggle.
* Ran the laptop (CPU) training and evaluation locally together with the author.

## Reused libraries and weights
PyTorch, Hugging Face Transformers / Hub / PEFT (training and reference checks), safetensors, FastAPI / Uvicorn,
pandas, NumPy, matplotlib, psutil. Pre-trained weights: SmolLM2-1.7B/360M/135M-Instruct; Qwen2.5-0.5B-Instruct was
considered as a candidate only.

## Verified rather than assumed
* The from-scratch decoder reproduces the Hugging Face logits (`tests/test_minillama.py`).
* Cached = uncached greedy tokens, speculative = greedy target tokens, tiled = reference attention (`tests/`, `results/`).
* All reported numbers are produced by scripts from saved logs and predictions (`scripts/make_tables.py`), not typed by hand.
* No code or results were copied from the TrialGPT repository; only its public dataset is used.

## Responsibility
The author reviewed the submission and is responsible for it.
