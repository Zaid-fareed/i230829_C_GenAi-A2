# Design decisions, rejected options and observed difficulties (rubric A3)

## Constraints discovered at the start
* Hardware: Intel i5-1135G7 (4 cores), 16 GB RAM, **no NVIDIA GPU**, ~12 GB free disk, no LaTeX installation.
* Consequence: everything is fp32 on CPU; only sub-1B models can be fine-tuned in hours rather than days.
  A measured smoke test gave ≈ 127–150 s per optimizer step (16 examples) for the 360M model.

## Decisions
| Topic | Chosen | Alternatives considered | Why |
|---|---|---|---|
| Target model | SmolLM2-360M-Instruct | Qwen2.5-0.5B-Instruct; SmolLM2-1.7B; Qwen2.5-1.5B | 1.5–1.7B models would need roughly 4–5× the compute per step on this CPU (my estimate from parameter ratio: >12 h for 2 epochs) and ~6–7 GB just for fp32 weights. Qwen2.5-0.5B has a 151k vocabulary (3× larger embedding/softmax) and QKV bias, and the Qwen2.5 family has no sibling below 0.5B, so it offers no smaller draft model with the same tokenizer. SmolLM2 has a 135M sibling with an identical tokenizer. |
| Draft model | SmolLM2-135M-Instruct, LoRA-tuned on the same train split | un-tuned 135M; n-gram/prompt-lookup drafting | Un-tuned draft is also measured (ablation). Prompt-lookup would not exercise a second Transformer. |
| Adaptation | LoRA r=16, α=32 on all linear projections | full fine-tuning; QLoRA; prefix-tuning | Full FT stores AdamW states for 360M params (~4 GB) and updates all weights – slower and prone to overfit 602 examples. QLoRA needs CUDA. |
| Output format | `{"label": …, "evidence": [ids]}` | free-text rationale; label-only classification head | Short JSON gives a parseable, auditable schema and keeps decode cost low. Free-text rationale would require explanation targets; the only ones in the data are GPT-4 outputs, which the assignment forbids. The app's explanation is therefore a deterministic *extractive* template over the model's label and cited sentences. |
| Decoder implementation | own `MiniLlama` reading HF safetensors | `model.generate(use_cache=True)`; HF `DynamicCache` | The brief requires explicit cache/position/mask handling and a pluggable attention kernel. Verified against HF logits (max |Δ| 6e-5). |
| KV cache layout | contiguous pre-allocated buffers, doubling growth, `truncate(n)` for rollback | `torch.cat` per step; paged cache | `cat` copies O(T) every token; paged caches matter for batching, not batch size 1. Rollback is a pointer move, which is what speculative decoding needs. |
| Precision | fp32 | bf16 | CPU bf16 matmul is slower on this CPU and would add numerical noise to equality tests. |
| Model selection | best validation macro-F1 epoch | last epoch; lowest val loss | Macro-F1 is the reported metric and the class distribution is skewed. |
| Overall verdict | explicit rule (any failed criterion → likely not eligible, …) | learned aggregator | No patient-level labels exist, so a learned aggregator cannot be validated; the rule is documented in the UI and report. |

## Difficulties (observed, not hypothetical)
1. **Missing criterion text** – annotation 883 has `criterion_text = NaN`. Kept in the split counts (integrity of the
   53-patient partition) but excluded from training/evaluation, recorded in the manifest.
2. **No `excluded` class in the test split** and only 2 in validation / 8 in training (class imbalance: 62.6 % of test
   rows are covered by the majority label of each criterion type). Hence macro-F1 is computed over classes present in gold or predictions and
   per-class tables are reported with support.
3. **Training speed** – see constraints; forced 2 epochs for the 360M target.
4. **Python/pip split** – the shell's default Python was an MSYS build without pip; all runs use the Windows Python 3.13.
5. **LaTeX unavailable offline** – the paper is delivered as `.tex` source; compile with Overleaf or a local TeX Live.
6. Peak memory on CPU: PyTorch has no CPU peak-allocation counter → sampled process RSS in a fresh subprocess per point
   (documented as an approximation).

## Addendum: second (GPU) configuration
| Topic | Chosen | Why |
|---|---|---|
| Headline model | SmolLM2-1.7B-Instruct + LoRA on a free Kaggle T4 | Same family/tokenizer as the CPU model; 4.7× more parameters; trains in ~19 min. A 7–8B model would need 4-bit CUDA-only libraries (QLoRA) and was out of scope. |
| Draft for the 1.7B | LoRA-tuned SmolLM2-360M | Only same-tokenizer model that is meaningfully smaller. Result: speculative decoding is exact but **slower** (0.39–0.81×) because the 360M draft has 32 layers vs 24 in the 1.7B target, so a draft step costs about as much as a target step on a launch-bound GPU. Documented, not hidden. |
| Precision | fp16 autocast for training, fp32 for all inference/evaluation/benchmarks | Keeps greedy-equality tests meaningful. |
| Two configurations | GPU = headline numbers, CPU = laptop/app model | Same split, prompts, metrics and code; only model pair/device/schedule differ. |

### Kaggle difficulties (observed)
1. Dataset mounted under `/kaggle/input/datasets/<user>/genai-a2`, not `/kaggle/input/genai-a2` → notebook now searches recursively.
2. `peft` refused to run because the preinstalled `torchao` 0.10.0 was too old → `pip uninstall torchao` (unused).
3. CUDA out-of-memory with micro-batch 4 on the 1.7B fp32 weights (14.5 GB in use) → micro-batch 1 × 16 accumulation (same effective batch).
4. `pip install -U transformers` printed unrelated dependency warnings (`datasets`, `gradio`); harmless.

### Findings that shaped the write-up
* Exclusion criteria: the fine-tuned 1.7B model answers `not excluded` for 130/131 test rows (majority-class collapse); all of the macro-F1 gain is from inclusion criteria. Reported as is.
* 40 of 42 "false decisions" are exclusion criteria with missing information answered `not excluded` — the failure mode the brief warns about; the app flags uncited decisive answers.
* Accuracy CI (0.578–0.818) overlaps the majority baseline (0.626); macro-F1 CI (0.359–0.554) is above 0.272. Claimed accordingly.
