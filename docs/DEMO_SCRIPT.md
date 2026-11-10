# Demo video script (target 6:00, allowed 5:00–7:00)

Legend: **DO** = what to click/type · **POINT** = what to circle or underline with the pen · **SAY** = narration
(if you record silently, still do the DO/POINT steps and pause 2–3 s on every POINT).

---

## 0. Preparation (before pressing record, ~5 min)
1. Close heavy programs. Make sure the app is **not** running yet (no black `run_app.bat` window open).
2. Open these browser tabs, in this order:

    - Tab 2: https://github.com/Zaid-fareed/i230829_C_GenAi-A2
    - Tab 3: https://huggingface.co/ZaidFareed/i230829-genai-a2-adapters

3. Open `docs/DEMO_SCRIPT.md` (this file) on a second screen or print it — you will copy the unseen case from section 3.
4. Open File Explorer at `Desktop\Semester 7\Gen AI\assignments\2` so `run_app.bat` is visible.
5. Loom: record **full screen** (not one tab), camera optional, microphone on if narrating.
   Check your Loom plan's maximum length — the video must be **at least 5 minutes**.

---

## 1. Start the app — 0:00 → 0:45
**DO** start recording. Show File Explorer. Double-click **`run_app.bat`**.
**POINT** the black window line `Using Python: ...Python313\python.exe`, then the line "wait until the badge says model ready".
**SAY** "This is my Gen AI Assignment 2: criterion-level clinical-trial eligibility screening. The whole system runs in the browser — one double-click on run_app.bat starts it, no IDE or notebook needed. It loads our fine-tuned SmolLM2 model."

**DO** when the browser opens, wait for the badge (cut the waiting later in Loom/Clipchamp if it is long).
**POINT** the green **model ready** badge, then the text next to it "SmolLM2-360M-Instruct + LoRA (target), SmolLM2-135M-Instruct + LoRA (draft)".
**POINT** the yellow banner "Educational screening prototype - not a clinical enrollment decision".
**SAY** "The header shows the loaded model and adapter. This banner is on every page: it is an educational prototype, not a medical decision tool."

---

## 2. Screen a held-out patient — 0:45 → 2:00
**DO** On **Screen a patient**, open the first dropdown "Load a held-out sample patient…" → choose **sigir-201410**.
**DO** Open the **Trial criteria** dropdown → choose **NCT02097186 - Preconditioning Shields Against Vascular Events in Surgery**.
**DO** Delete criteria with **×** until only these 5 remain:
`Age greater than 18 years` · `Patient willing to give full informed consent for participation` · `Patients undergoing elective carotid endarterectomy` · `Significant upper limb peripheral arterial disease` · `Previous history of upper limb deep vein thrombosis`.
**SAY** "This patient is from the held-out test split — the model never saw it during training. For each inclusion and exclusion criterion the model must give a label and cite the supporting sentences of the note."

**DO** Press **Screen patient** (about 15–30 s).
**POINT** the progress bar "Criterion 2/5…".
**POINT** on a result card: the coloured **label badge**, the **cited sentence** in the blue box, and "**dataset expert label: included ✓**".
**DO** Click the card "Patient willing to give full informed consent…".
**POINT** the yellow highlighted sentence **0.** in "Numbered patient note" at the bottom.
**SAY** "Each result shows the label, the exact sentence it is based on — highlighted in the note — and an explanation that only uses that evidence. For demo patients you also see the expert label, so you can see where the model is right or wrong."
**POINT** the card "Age greater than 18 years" (model says *not enough information*, expert says *included* ✗).
**SAY** "Here it missed that a 67-year-old is over 18 — the report analyses errors like this."
**POINT** the **Preliminary outcome** box at the top of Results and its rule text.
**SAY** "The overall outcome is an explicit rule over the criterion labels, not a learned verdict."

---

## 3. Unseen patient typed by me — 2:00 → 3:05
**DO** Click inside the **Patient note** box, select all (Ctrl+A), paste:
```
A 45-year-old man presents with a 3-day history of productive cough and fever of 38.9 C. He has type 2 diabetes treated with metformin. Chest X-ray shows a right lower lobe infiltrate. He reports no drug allergies. He smokes one pack of cigarettes per day.
```
**DO** Click **Preview sentence numbering**. **POINT** "5 sentences (auto-numbered)".
**SAY** "Now a completely new patient that is not in the dataset. Plain text is numbered into sentences automatically."
**DO** Click **Paste eligibility text** (to expand it), paste:
```
Inclusion Criteria:
* Adults aged 18 to 65 years
* Community-acquired pneumonia confirmed on chest radiograph
Exclusion Criteria:
* Pregnancy
* Known allergy to penicillin
* Current smoker
```
**DO** Click **Split into criteria**. **POINT** the five rows (2 inclusion, 3 exclusion).
**DO** Press **Screen patient**.
**POINT** each result as it appears; especially **Current smoker** and the sentence "He smokes one pack…".
**SAY** "Correct answers would be: included, included, not applicable, not excluded, and excluded for smoking."
- If smoker = **excluded**: "It correctly found the smoking sentence."
- If smoker = **not excluded**: "Here it fails — on exclusion criteria the model tends to answer 'not excluded'. This is the main weakness I measured and discuss in the report."

---

## 4. Invalid input and recovery — 3:05 → 3:30
**DO** Clear the patient note (Ctrl+A, Delete). Press **Screen patient**.
**POINT** the red message "Please enter a patient note (at least 20 characters)."
**SAY** "Invalid input is rejected with a clear message instead of crashing."
**DO** Choose **sigir-201410** again from the patient dropdown (note is refilled, message disappears).
**SAY** "And the app recovers immediately."

---

## 5. Report export — 3:30 → 4:00
**DO** Scroll to Results → **Download report (HTML)** (the results from section 3 are still there if you skipped re-running; otherwise re-run the 5 criteria quickly first). Open the downloaded file.
**POINT** in the report: the yellow disclaimer, "Generated (UTC)", "Model", "adapter sha256", "Decoding", then one criterion block.
**SAY** "The report can be downloaded as HTML, Markdown or JSON, with provenance: model, adapter hash, decoding settings and time."
**DO** Return to the app tab.

---

## 6. Decoding comparison — 4:00 → 5:05
**DO** Click the **Compare decoding** tab. Leave all four boxes ticked, repeats = 1, k = 4, force output length = 0. Press **Run comparison** (about 20–30 s).
**SAY** "This runs the same input through the four decoding methods I implemented, live on this laptop."
**POINT** the **tokens match** column (all "✓ identical").
**SAY** "All four give exactly the same output, so the optimisations do not change the answer."
**POINT** the **total s** of `uncached` vs `kv_cache`, and the latency bars.
**SAY** "Without a cache the model recomputes the whole text for every new word; with my KV cache it reuses earlier keys and values — several times faster."
**POINT** `kv_cache_tiled`.
**SAY** "This is my tiled attention with online softmax: numerically the same, but slower in plain PyTorch than the fused kernel."
**POINT** `speculative` row: **target passes** and **acceptance**.
**SAY** "Speculative decoding: the small draft model proposes words and the big model checks them in one pass — fewer target passes, same output."

---

## 7. Stored results — 5:05 → 5:45
**DO** Click the **Experiments** tab. Scroll slowly.
**POINT** under "Configuration A - GPU": the table rows **base, zero-shot 0.146** and **fine-tuned (LoRA) 0.709**, and **majority label 0.626**.
**SAY** "On the held-out test set the 1.7-billion-parameter model trained on a Kaggle GPU goes from 15% to 71% accuracy; macro-F1 from 0.11 to 0.46. Always guessing the common label gives 63%, which the report discusses honestly."
**POINT** the KV table title "tokens identical to reference in **27/27** runs".
**POINT** attention table **max |err|** column (~1e-6) and the **peak Δ MB** at 4096: reference **2956** vs tiled **45**.
**SAY** "The cache was token-identical in all 27 tests; tiled attention matches the reference to about one millionth and uses far less memory."
**POINT** speculative table "**32/32**" and the speed-up column (below 1).
**SAY** "Speculative decoding was exact on all 32 prompts but slower on the GPU, because the draft model is as deep as the target — explained in the report."

---

## 8. Code and weights — 5:45 → 6:15
**DO** Switch to the GitHub tab. Scroll the README "Results at a glance".
**POINT** folders `src/`, `app/`, `report/`, and the README results table.
**SAY** "Everything is on GitHub: code, the fixed 37/8/8 patient split, results, logs and reproduction steps."
**DO** Switch to the Hugging Face tab. **POINT** the four adapter folders.
**SAY** "The trained adapters are on Hugging Face and can be downloaded with one script. Thank you."
**DO** Stop recording.

---

## After recording
1. Trim the model-loading wait if needed; check the length is between 5:00 and 7:00.
2. Upload / share as **unlisted** (YouTube) or with link access (Loom), and test the link in a private window.
3. Paste the link into Overleaf `main.tex` at `PASTE-YOUTUBE-LINK-HERE`, recompile, send the PDF and link.
