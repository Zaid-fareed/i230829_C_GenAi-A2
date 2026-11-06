# Research notes (rubric A1–A3)

Sources I studied and what each one changed in the design. All summaries are in my own words.
Citation keys match `report/references.bib`. **Bibliographic details were written from memory of the
papers – re-check each entry (volume, pages, arXiv number) against the publisher page before submitting.**

| # | Source | What I took from it | Decision it influenced |
|---|--------|---------------------|------------------------|
| 1 | Jin et al., *Matching patients to clinical trials with large language models*, Nat. Commun. 2024 (`jin2024trialgpt`) | Defines the three-stage TrialGPT pipeline and the criterion-level annotation set: label space per criterion type, evidence sentence IDs, GPT-4 as the reference model. | Label sets, sentence-ID evidence format, rule-based aggregation (they aggregate criterion scores; I use explicit rules). GPT-4 columns are removed so nothing leaks. |
| 2 | Koopman & Zuccon, *A test collection for matching patients to clinical trials*, SIGIR 2016 (`koopman2016sigir`) | Origin of the short, synthetic patient descriptions (sigir-2014/2015 IDs). | Explains why notes are short (~150 tokens) → no truncation needed, CPU training feasible. |
| 3 | Hu et al., *LoRA*, ICLR 2022 (`hu2022lora`) | Freeze W, learn ΔW = (α/r)·BA. Memory/optimizer states only for adapters. | Chosen adaptation method on a 16 GB CPU laptop; adapter is merged by hand inside my own decoder. |
| 4 | Dettmers et al., *QLoRA*, NeurIPS 2023 (`dettmers2023qlora`) | 4-bit base weights + LoRA. | **Rejected**: needs bitsandbytes/CUDA kernels; no GPU available; at 360M parameters fp32 fits easily. |
| 5 | Vaswani et al., *Attention is all you need*, NeurIPS 2017 (`vaswani2017attention`) | Scaled dot-product attention, causal masking. | Reference attention implementation (§E1). |
| 6 | Milakov & Gimelshein, *Online normalizer calculation for softmax*, 2018 (`milakov2018online`) | One-pass running max/normalizer recurrence. | Core recurrence in my tiled attention. |
| 7 | Dao et al., *FlashAttention*, NeurIPS 2022 (`dao2022flashattention`); Rabe & Staats 2021 (`rabe2021selfattention`) | Tiling to avoid materialising the T×T matrix; memory O(T) instead of O(T²). | Tiled kernel structure (outer loop on query blocks, inner on key blocks, causal block skipping). Also explains why a Python version is slower than the fused kernel. |
| 8 | Su et al., *RoFormer*, 2021/2024 (`su2024roformer`) | Rotary position embedding rotates q,k by absolute position. | Position handling in the KV cache: K is cached *after* rotation, new tokens get positions `cache.length + i`. |
| 9 | Ainslie et al., *GQA*, EMNLP 2023 (`ainslie2023gqa`); Shazeer 2019 (`shazeer2019mqa`) | Fewer KV heads shrink the KV cache. | SmolLM2 uses 5 KV heads for 15 query heads → cache is 3× smaller; my attention broadcasts KV heads lazily per block. |
| 10 | Leviathan et al., ICML 2023 (`leviathan2023speculative`); Chen et al. 2023 (`chen2023speculative`) | Draft-then-verify; exactness guarantees; speed-up depends on acceptance rate α and draft cost c. | Greedy variant: accept longest matching prefix, take target's token at the first mismatch. Predicts when slowdown happens (draft not cheap enough). |
| 11 | Allal et al., *SmolLM2*, 2025 (`allal2025smollm2`) | 135M/360M/1.7B Llama-style models, Apache-2.0, shared tokenizer, 8k context. | Target = 360M, draft = 135M (same tokenizer → drafted token ids are directly comparable). |
| 12 | Qwen Team, *Qwen2.5 technical report*, 2024 (`qwen2024qwen25`) | 0.5B model, 32k context, Apache-2.0, 151k-token vocabulary. | Second candidate in the model comparison (measured, see `results/metrics/candidate_compare.json`). |
| 13 | Pope et al., *Efficiently scaling Transformer inference*, MLSys 2023 (`pope2023efficiently`); Kwon et al., *PagedAttention*, SOSP 2023 (`kwon2023pagedattention`) | KV cache memory growth and management strategies. | I use contiguous pre-allocated buffers with doubling; paged caches rejected as unnecessary for batch size 1. |
| 14 | Touvron et al., *LLaMA* (`touvron2023llama`), Zhang & Sennrich RMSNorm (`zhang2019rmsnorm`), Shazeer GLU variants (`shazeer2020glu`) | Components of the Llama block. | My `MiniLlama` re-implements RMSNorm, SwiGLU MLP, RoPE, GQA. |
| 15 | Wolf et al., *Transformers* (`wolf2020transformers`) | Reference implementation. | Used to verify my decoder (logit agreement) and for assisted-generation cross-check. |

## Candidate models

| | SmolLM2-360M-Instruct | SmolLM2-135M-Instruct | Qwen2.5-0.5B-Instruct |
|---|---|---|---|
| Architecture | Llama, 32 layers, d=960, 15 heads / 5 KV | Llama, 30 layers, d=576, 9 heads / 3 KV | Qwen2, 24 layers, d=896, 14 heads / 2 KV, QKV bias |
| Context | 8192 | 8192 | 32768 |
| Vocabulary | 49,152 | 49,152 | 151,936 |
| License | Apache-2.0 | Apache-2.0 | Apache-2.0 |
| Role | **target** | **draft** | rejected (see design decisions) |

Measured numbers (parameter counts, one LoRA step time, zero-shot behaviour) are produced by
`scripts/candidate_compare.py` → `results/metrics/candidate_compare.json`.
