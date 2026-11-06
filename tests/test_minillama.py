"""MiniLlama vs Hugging Face, cached vs uncached greedy decoding, speculative equality.

Uses the small 135M model so the tests stay fast on CPU.  Run: python tests/test_minillama.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from minillama import MiniLlama  # noqa: E402
from decode import generate_uncached, generate_cached, generate_speculative  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

BASE_S = ROOT / "checkpoints" / "base" / "SmolLM2-135M-Instruct"
BASE_T = ROOT / "checkpoints" / "base" / "SmolLM2-360M-Instruct"
torch.set_num_threads(2)

TEXT = ("The patient is a 58-year-old woman with hypertension and obesity who presents with chest pain "
        "that started two days ago while walking. She denies diabetes and smoking. ") * 3


def test_logits_match_hf():
    tok = AutoTokenizer.from_pretrained(BASE_S)
    ids = torch.tensor([tok(TEXT[:200])["input_ids"]])
    hf = AutoModelForCausalLM.from_pretrained(BASE_S, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        ref = hf(ids).logits
    for name in ("reference", "tiled", "sdpa"):
        m = MiniLlama(BASE_S, attn=name)
        out = m.forward(ids)
        diff = (out - ref).abs().max().item()
        print(f"  logits max|diff| vs HF [{name}] = {diff:.2e}")
        assert diff < 1e-3
        assert torch.equal(out.argmax(-1), ref.argmax(-1))


def test_cached_equals_uncached_three_lengths():
    tok = AutoTokenizer.from_pretrained(BASE_S)
    m = MiniLlama(BASE_S, attn="sdpa")
    for n_in, n_out in [(16, 12), (64, 24), (150, 40)]:
        prompt = tok(TEXT)["input_ids"][:n_in]
        a = generate_uncached(m, prompt, n_out, eos_ids=(), min_new=n_out)
        b = generate_cached(m, prompt, n_out, eos_ids=(), min_new=n_out)
        print(f"  in={n_in} out={n_out}: equal={a.tokens == b.tokens}")
        assert a.tokens == b.tokens


def test_cached_with_every_attention_backend():
    tok = AutoTokenizer.from_pretrained(BASE_S)
    prompt = tok(TEXT)["input_ids"][:90]
    outs = {}
    for name in ("reference", "tiled", "sdpa"):
        m = MiniLlama(BASE_S, attn=name)
        outs[name] = generate_cached(m, prompt, 30, min_new=30).tokens
    assert outs["reference"] == outs["tiled"] == outs["sdpa"]


def test_cache_truncate_and_growth():
    tok = AutoTokenizer.from_pretrained(BASE_S)
    m = MiniLlama(BASE_S)
    prompt = tok(TEXT)["input_ids"][:40]
    cache = m.new_cache(1, capacity=16)          # force several doublings
    lg1 = m.forward(torch.tensor([prompt]), cache=cache, last_only=True)
    assert cache.n_grow >= 1 and cache.length == 40
    cache.truncate(30)
    lg2 = m.forward(torch.tensor([prompt[30:]]), cache=cache, last_only=True)   # re-feed the last 10
    assert torch.allclose(lg1, lg2, atol=1e-4)


def test_speculative_equals_greedy_target():
    tok = AutoTokenizer.from_pretrained(BASE_S)
    target, draft = MiniLlama(BASE_T), MiniLlama(BASE_S)
    for n_in, n_out, k in [(30, 24, 3), (80, 40, 4), (120, 30, 6)]:
        prompt = tok(TEXT)["input_ids"][:n_in]
        ref = generate_cached(target, prompt, n_out, min_new=n_out)
        sp = generate_speculative(target, draft, prompt, n_out, k=k, min_new=n_out)
        print(f"  in={n_in} out={n_out} k={k}: equal={ref.tokens == sp.tokens} acc={sp.stats['acceptance_rate']:.2f}")
        assert ref.tokens == sp.tokens


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name); fn(); print("  ok")
