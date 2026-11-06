"""Criterion-level screening pipeline shared by evaluation scripts and the web app."""
from __future__ import annotations

import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

import data as D
from decode import generate_cached, generate_speculative, generate_uncached
from minillama import MiniLlama

ROOT = Path(__file__).resolve().parents[1]
import json
import os

# A "profile" selects the model pair and device.  Default = the CPU profile used for the laptop results;
# set TRIAL_PROFILE=configs/profile_gpu.json (Kaggle) to use the larger GPU pair.
PROFILE_PATH = Path(os.environ.get("TRIAL_PROFILE", ROOT / "configs" / "profile_cpu.json"))
PROFILE = json.loads(Path(PROFILE_PATH).read_text())
TARGET_BASE = ROOT / PROFILE["target_base"]
DRAFT_BASE = ROOT / PROFILE["draft_base"]
TARGET_ADAPTER = ROOT / PROFILE["target_adapter"]
DRAFT_ADAPTER = ROOT / PROFILE["draft_adapter"]
DEVICE = PROFILE.get("device", "cpu")
if DEVICE == "cuda" and not torch.cuda.is_available():
    DEVICE = "cpu"
RESULT_TAG = PROFILE.get("result_dir", "metrics")        # results/<RESULT_TAG>/

MODES = {
    "uncached": "Greedy, no KV cache (reference: full recompute at every step)",
    "kv_cache": "Greedy with explicit KV cache (PyTorch SDPA attention)",
    "kv_cache_tiled": "Greedy with explicit KV cache + tiled online-softmax attention",
    "speculative": "Greedy speculative decoding (135M draft, 360M target, KV caches)",
}
MAX_NEW = 48
MAX_PROMPT_TOKENS = 3000


class Engine:
    def __init__(self, target_adapter=TARGET_ADAPTER, draft_adapter=DRAFT_ADAPTER, load_draft=True, k=4):
        self.tok = AutoTokenizer.from_pretrained(TARGET_BASE)
        self.target = MiniLlama(TARGET_BASE, attn="sdpa", device=DEVICE)
        self.adapter_name = "base (no adapter)"
        if target_adapter and Path(target_adapter).exists():
            self.target.merge_lora(target_adapter)
            self.adapter_name = str(Path(target_adapter).parent.name)
        self.draft = None
        if load_draft:
            self.draft = MiniLlama(DRAFT_BASE, attn="sdpa", device=DEVICE)
            if draft_adapter and Path(draft_adapter).exists():
                self.draft.merge_lora(draft_adapter)
        self.k = k
        self.eos = tuple({self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.eos_token_id})

    # -- prompt ---------------------------------------------------------------------------
    def encode(self, note, ctype, ctext):
        msgs = D.build_messages(note, ctype, ctext)
        text = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return self.tok(text, add_special_tokens=False)["input_ids"]

    # -- raw generation under a given decoding mode -----------------------------------------
    def generate(self, ids, mode="kv_cache", max_new=MAX_NEW, min_new=0, k=None):
        if mode == "uncached":
            self.target.set_attention("sdpa")
            return generate_uncached(self.target, ids, max_new, self.eos, min_new)
        if mode == "kv_cache":
            self.target.set_attention("sdpa")
            return generate_cached(self.target, ids, max_new, self.eos, min_new)
        if mode == "kv_cache_tiled":
            self.target.set_attention("tiled")
            try:
                return generate_cached(self.target, ids, max_new, self.eos, min_new)
            finally:
                self.target.set_attention("sdpa")
        if mode == "speculative":
            assert self.draft is not None, "draft model not loaded"
            self.target.set_attention("sdpa"); self.draft.set_attention("sdpa")
            return generate_speculative(self.target, self.draft, ids, max_new, k or self.k, self.eos, min_new)
        raise ValueError(f"unknown mode {mode}")

    # -- full criterion-level prediction -----------------------------------------------------
    def predict(self, note, ctype, ctext, mode="kv_cache"):
        ids = self.encode(note, ctype, ctext)
        if len(ids) > MAX_PROMPT_TOKENS:
            raise ValueError(f"prompt has {len(ids)} tokens (limit {MAX_PROMPT_TOKENS}); shorten the note or criterion")
        t0 = time.perf_counter()
        res = self.generate(ids, mode)
        wall = time.perf_counter() - t0
        raw = self.tok.decode([t for t in res.tokens if t not in self.eos], skip_special_tokens=True)
        label, ev, ok = D.parse_output(raw, ctype)
        sents = D.split_sentences(note)
        valid_ev = [i for i in ev if i in sents]
        out = {
            "criterion_type": ctype, "criterion_text": ctext, "raw_output": raw,
            "output_well_formed": ok, "label": label,
            "evidence_ids": valid_ev, "dropped_unknown_ids": [i for i in ev if i not in sents],
            "evidence_text": [{"id": i, "text": sents[i]} for i in valid_ev],
            "explanation": explain(label, ctype, valid_ev, sents),
            "n_prompt_tokens": len(ids), "n_new_tokens": len(res.tokens), "mode": mode,
            "latency_s": wall, "prefill_s": res.prefill_s, "decode_s": res.decode_s,
            "tokens_per_s": res.tok_per_s, "stats": {k_: v for k_, v in res.stats.items() if k_ != "accepted_hist"},
        }
        return out


def explain(label, ctype, ev_ids, sents):
    """Deterministic, extractive explanation that only restates the model's label and the cited text."""
    cited = "; ".join(f"[{i}] \"{sents[i]}\"" for i in ev_ids)
    if label == "invalid":
        return "The model did not return a valid label, so no conclusion is drawn for this criterion."
    if label == "not enough information":
        base = "The note does not contain enough information to decide this criterion."
        return base + (f" Closest cited sentences: {cited}. Missing information is not treated as evidence of eligibility or ineligibility." if cited
                       else " No note sentence is cited; absence of a mention is not treated as evidence either way.")
    if label == "not applicable":
        return "The criterion was judged not applicable to this patient." + (f" Cited: {cited}." if cited else "")
    positive = {"included": "meets", "not excluded": "does not trigger"}
    negative = {"not included": "does not meet", "excluded": "triggers"}
    verb = positive.get(label) or negative.get(label)
    head = f"Predicted '{label}': the patient {verb} this {ctype} criterion"
    return head + (f", based on: {cited}." if cited else ". No supporting sentence was cited, so this rests on the model's judgement of the whole note and should be checked manually.")
