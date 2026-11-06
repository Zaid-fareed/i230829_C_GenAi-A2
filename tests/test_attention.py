"""Numerical tests for the attention back-ends.  Run: python -m pytest tests -q  (or python tests/test_attention.py)"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from attention import reference_attention, tiled_attention, sdpa_attention  # noqa: E402

torch.manual_seed(0)
TOL = 1e-5


def rand_qkv(B, H, Hkv, Tq, Tk, D, scale=1.0):
    q = torch.randn(B, H, Tq, D) * scale
    k = torch.randn(B, Hkv, Tk, D) * scale
    v = torch.randn(B, Hkv, Tk, D)
    return q, k, v


def test_hand_computed_two_token_example():
    # One head, D=1.  q=[1,1], k=[0,ln2], v=[10,20]; causal, scale = 1/sqrt(1) = 1.
    q = torch.tensor([[[[1.0], [1.0]]]])
    k = torch.tensor([[[[0.0], [math.log(2.0)]]]])
    v = torch.tensor([[[[10.0], [20.0]]]])
    out = reference_attention(q, k, v)[0, 0, :, 0]
    # row 0 sees only key 0 -> 10 ; row 1 sees both: weights 1/3, 2/3 -> 10/3 + 40/3 = 50/3
    assert torch.allclose(out, torch.tensor([10.0, 50.0 / 3.0]), atol=1e-6)
    for fn in (tiled_attention, sdpa_attention):
        assert torch.allclose(fn(q, k, v)[0, 0, :, 0], out, atol=1e-6)


def test_tiled_matches_reference_many_shapes():
    worst = 0.0
    for (B, H, Hkv, Tq, D, bq, bk) in [(1, 2, 2, 5, 8, 2, 3), (2, 4, 2, 17, 16, 4, 5), (1, 6, 3, 33, 32, 8, 16),
                                       (1, 4, 4, 64, 64, 64, 64), (2, 3, 1, 50, 8, 7, 11), (1, 1, 1, 1, 8, 1, 1)]:
        q, k, v = rand_qkv(B, H, Hkv, Tq, Tq, D)
        ref = reference_attention(q, k, v)
        til = tiled_attention(q, k, v, block_q=bq, block_k=bk)
        worst = max(worst, (ref - til).abs().max().item())
        assert torch.allclose(ref, til, atol=TOL), (B, H, Tq, bq, bk)
    assert worst < TOL


def test_cached_decode_shapes_with_offset():
    # Tq new tokens appended after a cache of 21 tokens (q_offset = 21)
    for Tq in (1, 3, 8):
        q, k, v = rand_qkv(1, 4, 2, Tq, 21 + Tq, 16)
        ref = reference_attention(q, k, v, q_offset=21)
        til = tiled_attention(q, k, v, q_offset=21, block_q=4, block_k=6)
        assert torch.allclose(ref, til, atol=TOL)
        assert torch.allclose(ref, sdpa_attention(q, k, v, q_offset=21), atol=TOL)


def test_causality_no_future_leak():
    q, k, v = rand_qkv(1, 2, 2, 10, 10, 8)
    base = tiled_attention(q, k, v, block_q=3, block_k=3)
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 7:] += 100.0          # perturb the future of positions < 7
    v2[:, :, 7:] -= 50.0
    pert = tiled_attention(q, k2, v2, block_q=3, block_k=3)
    assert torch.allclose(base[:, :, :7], pert[:, :, :7], atol=1e-6)


def test_numerical_stability_large_scores():
    q, k, v = rand_qkv(1, 2, 2, 12, 12, 16, scale=40.0)   # scores of magnitude ~ 1e3
    ref = reference_attention(q, k, v)
    til = tiled_attention(q, k, v, block_q=4, block_k=4)
    assert torch.isfinite(ref).all() and torch.isfinite(til).all()
    assert torch.allclose(ref, til, atol=1e-4)
    # a naive softmax without max-subtraction overflows in float32
    s = (q @ k.transpose(-1, -2)) / 4.0
    assert not torch.isfinite(torch.exp(s * 10)).all()


def test_padding_key_mask():
    q, k, v = rand_qkv(2, 2, 2, 6, 6, 8)
    km = torch.tensor([[True] * 6, [False, False] + [True] * 4])   # left-padded second sequence
    ref = reference_attention(q, k, v, key_mask=km)
    til = tiled_attention(q, k, v, key_mask=km, block_q=2, block_k=2)
    assert torch.allclose(ref[0], til[0], atol=TOL)
    assert torch.allclose(ref[1, :, 2:], til[1, :, 2:], atol=TOL)    # padded query rows are undefined


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
