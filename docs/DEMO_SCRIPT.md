# 5–7 minute demo video script

Record the screen with voice-over (OBS, Xbox Game Bar `Win+G`, or Zoom). Start the app first (`run_app.bat`) and wait for the green "model ready" pill.
Target length ≈ 6:30. Times are a guide.

| Time | Screen | Say / do |
|---|---|---|
| 0:00–0:30 | Repo folder tree + app header banner | Task, dataset (TrialGPT criterion annotations, 37/8/8 patient split), model (SmolLM2-360M + LoRA), prototype disclaimer. |
| 0:30–1:30 | **Screen** tab | Load a held-out sample patient → pick a sample trial → *Screen patient* (kv_cache). Point at labels, highlighted cited sentences, the "dataset expert label ✓/✗" comparison, the overall outcome rule. |
| 1:30–2:30 | Unseen case | Paste your own short note and a pasted "Inclusion/Exclusion Criteria" block (*Split into criteria*). Show a *not enough information* result and say that missing mention is not treated as evidence. |
| 2:30–3:00 | Error handling + export | Click *Screen* with an empty note → error message. Enter a 1-word criterion → rejected. Download the HTML report and open it (disclaimer, provenance, evidence). |
| 3:00–4:30 | **Compare decoding** tab | Run all four modes. Explain: uncached recomputes everything (slowest); KV cache feeds one token; tiled attention is slower in PyTorch (Python loops) but numerically identical; speculative shows acceptance rate and whether it was faster. Point out "tokens match: identical". Repeat with *force output length* = 64. |
| 4:30–5:30 | **Experiments** tab + `results/` | Before/after table, KV benchmark chart, attention chart, speculative table. Mention confusion matrices and error analysis in the paper. |
| 5:30–6:30 | Code (`minillama.py`, `attention.py`, `decode.py`) | Show `KVCache.write/truncate`, position offset in `forward`, the online-softmax update lines, and the speculative cache rollback (`tc.truncate`, draft top-up). Show test output (`python tests/test_minillama.py`). |
| 6:30–7:00 | Paper PDF first page | Summarise results honestly: what worked, what did not (e.g. slowdowns), limitations. |
