"""Scaled dot-product attention back-ends.

All back-ends share one signature::

    fn(q, k, v, q_offset=0, causal=True, key_mask=None) -> (B, H, Tq, D)

q          : (B, H,   Tq, D)   queries of the *new* tokens
k, v       : (B, Hkv, Tk, D)   keys / values of every visible token (cache + new)
q_offset   : absolute position of q[..., 0, :]; with a KV cache this equals the
             number of tokens already cached, so query i may see keys j <= q_offset + i
key_mask   : optional (B, Tk) bool, True = real token (used for padded batches)

H must be a multiple of Hkv (grouped-query attention); K/V are broadcast to the
query heads lazily, per block, so the tiled kernel never materialises a full
expanded copy.
"""
from __future__ import annotations

import math

import torch

NEG_INF = float("-inf")


def _expand_kv(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """(B, Hkv, T, D) -> (B, H, T, D) by repeating every KV head H/Hkv times."""
    b, hkv, t, d = x.shape
    if hkv == n_heads:
        return x
    rep = n_heads // hkv
    return x[:, :, None].expand(b, hkv, rep, t, d).reshape(b, n_heads, t, d)


def _causal_block_mask(q_start: int, q_len: int, k_start: int, k_len: int, q_offset: int, device):
    """Bool mask (q_len, k_len): True where query (abs pos q_offset+q_start+i) may see key k_start+j."""
    qi = torch.arange(q_len, device=device)[:, None] + (q_offset + q_start)
    kj = torch.arange(k_len, device=device)[None, :] + k_start
    return kj <= qi


# ---------------------------------------------------------------------------
# 1. Reference implementation: materialises the whole (Tq x Tk) score matrix.
# ---------------------------------------------------------------------------
def reference_attention(q, k, v, q_offset=0, causal=True, key_mask=None):
    n_heads, d = q.shape[1], q.shape[-1]
    k, v = _expand_kv(k, n_heads), _expand_kv(v, n_heads)
    scale = 1.0 / math.sqrt(d)
    s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale          # (B,H,Tq,Tk)
    mask = torch.ones(s.shape[-2:], dtype=torch.bool, device=s.device)
    if causal:
        mask = _causal_block_mask(0, q.shape[2], 0, k.shape[2], q_offset, s.device)
    s = s.masked_fill(~mask, NEG_INF)
    if key_mask is not None:
        s = s.masked_fill(~key_mask[:, None, None, :], NEG_INF)
    # numerically stable softmax: subtract the row maximum before exponentiating.
    m = s.amax(dim=-1, keepdim=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))                # fully-masked rows -> 0, not NaN
    e = torch.exp(s - m)
    denom = e.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    return torch.matmul(e / denom, v.float()).to(q.dtype)


# ---------------------------------------------------------------------------
# 2. Tiled attention with online softmax (FlashAttention-style recurrence).
# ---------------------------------------------------------------------------
def tiled_attention(q, k, v, q_offset=0, causal=True, key_mask=None, block_q=64, block_k=64):
    """Blocked attention that never builds the full score matrix.

    For every query block we sweep over key blocks while keeping, per query row,
        m : running maximum of the scores seen so far,
        l : running sum of exp(score - m),
        o : running un-normalised output  sum_j exp(score_j - m) v_j.
    When a new key block arrives with block maximum m_b:
        m' = max(m, m_b);  alpha = exp(m - m');
        l' = alpha * l + sum exp(s_b - m');  o' = alpha * o + exp(s_b - m') @ v_b
    and the result is o / l after the last block.  Key blocks lying entirely above
    the causal diagonal are skipped.
    """
    B, H, Tq, D = q.shape
    Tk = k.shape[2]
    scale = 1.0 / math.sqrt(D)
    out = torch.empty(B, H, Tq, D, dtype=torch.float32, device=q.device)
    qf = q.float()
    for qs in range(0, Tq, block_q):
        qe = min(qs + block_q, Tq)
        q_blk = qf[:, :, qs:qe]
        m = torch.full((B, H, qe - qs, 1), NEG_INF, device=q.device)
        l = torch.zeros((B, H, qe - qs, 1), device=q.device)
        o = torch.zeros((B, H, qe - qs, D), device=q.device)
        # last key position visible to this query block
        k_hi = min(Tk, q_offset + qe) if causal else Tk
        for ks in range(0, k_hi, block_k):
            ke = min(ks + block_k, k_hi)
            k_blk = _expand_kv(k[:, :, ks:ke], H).float()
            v_blk = _expand_kv(v[:, :, ks:ke], H).float()
            s = torch.matmul(q_blk, k_blk.transpose(-1, -2)) * scale          # (B,H,bq,bk)
            if causal:
                cm = _causal_block_mask(qs, qe - qs, ks, ke - ks, q_offset, q.device)
                s = s.masked_fill(~cm, NEG_INF)
            if key_mask is not None:
                s = s.masked_fill(~key_mask[:, None, None, ks:ke], NEG_INF)
            m_blk = s.amax(dim=-1, keepdim=True)
            m_new = torch.maximum(m, m_blk)
            m_safe = torch.where(torch.isfinite(m_new), m_new, torch.zeros_like(m_new))
            alpha = torch.exp(torch.where(torch.isfinite(m), m, m_safe) - m_safe)   # rescales old stats
            alpha = torch.where(torch.isfinite(m), alpha, torch.zeros_like(alpha))
            p = torch.exp(s - m_safe)
            l = alpha * l + p.sum(dim=-1, keepdim=True)
            o = alpha * o + torch.matmul(p, v_blk)
            m = m_new
        out[:, :, qs:qe] = o / l.clamp_min(1e-30)
    return out.to(q.dtype)


# ---------------------------------------------------------------------------
# 3. PyTorch's fused scaled_dot_product_attention (for comparison only).
# ---------------------------------------------------------------------------
def sdpa_attention(q, k, v, q_offset=0, causal=True, key_mask=None):
    H = q.shape[1]
    k, v = _expand_kv(k, H), _expand_kv(v, H)
    mask = None
    if causal:
        mask = _causal_block_mask(0, q.shape[2], 0, k.shape[2], q_offset, q.device)[None, None]
    if key_mask is not None:
        km = key_mask[:, None, None, :]
        mask = km if mask is None else (mask & km)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)


BACKENDS = {
    "reference": reference_attention,
    "tiled": tiled_attention,
    "sdpa": sdpa_attention,
}
