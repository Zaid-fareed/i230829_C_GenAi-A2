"""Greedy decoding loops: uncached reference, KV-cached, and speculative.

Every function takes a list of prompt token ids and returns a ``GenResult``.
``min_new`` lets benchmarks suppress EOS for the first n tokens so that output
length can be controlled identically across paths.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from minillama import MiniLlama


@dataclass
class GenResult:
    tokens: list[int]
    prefill_s: float = 0.0
    decode_s: float = 0.0
    n_forward_target: int = 0
    n_forward_draft: int = 0
    stats: dict = field(default_factory=dict)

    @property
    def total_s(self):
        return self.prefill_s + self.decode_s

    @property
    def tok_per_s(self):
        return len(self.tokens) / self.total_s if self.total_s > 0 else float("nan")

    @property
    def decode_tok_per_s(self):
        n = max(len(self.tokens) - 1, 0)       # first token comes out of prefill
        return n / self.decode_s if self.decode_s > 0 else float("nan")


def _sync(device):
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _pick(logits_row, eos_ids, allow_eos):
    """Greedy argmax; optionally forbid EOS tokens (used to force a minimum length)."""
    if not allow_eos:
        logits_row = logits_row.clone()
        logits_row[list(eos_ids)] = float("-inf")
    return int(torch.argmax(logits_row).item())


# ---------------------------------------------------------------------------
# Reference path: no cache.  Every step re-runs the full sequence from position 0.
# ---------------------------------------------------------------------------
def generate_uncached(model: MiniLlama, prompt, max_new, eos_ids=(), min_new=0) -> GenResult:
    dev = model.device
    seq = list(prompt)
    res = GenResult(tokens=[])
    for step in range(max_new):
        _sync(dev); t0 = time.perf_counter()
        ids = torch.tensor([seq], device=dev)
        logits = model.forward(ids, cache=None, last_only=True)
        nxt = _pick(logits[0, -1], eos_ids, step >= min_new)
        _sync(dev); dt = time.perf_counter() - t0
        if step == 0:
            res.prefill_s += dt
        else:
            res.decode_s += dt
        res.n_forward_target += 1
        seq.append(nxt)
        res.tokens.append(nxt)
        if nxt in eos_ids:
            break
    return res


# ---------------------------------------------------------------------------
# KV-cached path: prefill once, then feed ONE new token per step.
# ---------------------------------------------------------------------------
def generate_cached(model: MiniLlama, prompt, max_new, eos_ids=(), min_new=0, cache=None) -> GenResult:
    dev = model.device
    cache = cache or model.new_cache(1, capacity=max(256, len(prompt) + max_new + 8))
    res = GenResult(tokens=[])
    _sync(dev); t0 = time.perf_counter()
    logits = model.forward(torch.tensor([prompt], device=dev), cache=cache, last_only=True)   # prefill
    nxt = _pick(logits[0, -1], eos_ids, 0 >= min_new)
    _sync(dev); res.prefill_s = time.perf_counter() - t0
    res.n_forward_target += 1
    res.tokens.append(nxt)
    t1 = time.perf_counter()
    while len(res.tokens) < max_new and nxt not in eos_ids:
        # positions are implicit: the new token gets position cache.length (see MiniLlama.forward)
        logits = model.forward(torch.tensor([[nxt]], device=dev), cache=cache, last_only=True)
        nxt = _pick(logits[0, -1], eos_ids, len(res.tokens) >= min_new)
        res.tokens.append(nxt)
        res.n_forward_target += 1
    _sync(dev); res.decode_s = time.perf_counter() - t1
    res.stats["cache_len"] = cache.length
    res.stats["kv_bytes_used"] = cache.nbytes_used()
    res.stats["kv_grow_events"] = cache.n_grow
    return res


# ---------------------------------------------------------------------------
# Greedy speculative decoding (draft proposes k tokens, target verifies them in ONE pass).
# ---------------------------------------------------------------------------
def generate_speculative(target: MiniLlama, draft: MiniLlama, prompt, max_new, k=4, eos_ids=(), min_new=0) -> GenResult:
    """Invariant at the top of every round:
         * ``seq`` = prompt + accepted tokens, ``last`` = seq[-1]
         * both caches hold K/V for seq[:-1]  (i.e. everything except ``last``)
       Round:
         1. draft decodes k tokens autoregressively starting from ``last``  -> d1..dk
         2. target runs ONE forward on [last, d1..dk]; its logits at position i give the
            greedy token g_i that follows the i-th input
         3. accept the longest prefix with d_{i+1} == g_i; the first mismatch (or the bonus
            position after dk) contributes the target's own token g_n, so every round
            emits n+1 >= 1 tokens and the output is identical to greedy target decoding
         4. caches are rolled back to seq[:-1] of the new sequence (truncate), and the
            draft cache is topped up with d_k when all drafts were accepted.
    """
    dev = target.device
    tc = target.new_cache(1, capacity=len(prompt) + max_new + k + 8)
    dc = draft.new_cache(1, capacity=len(prompt) + max_new + k + 8)
    res = GenResult(tokens=[])
    proposed = accepted = rounds = 0
    accepted_hist = []

    _sync(dev); t0 = time.perf_counter()
    # prefill both with prompt[:-1]; the last prompt token starts the first round (keeps the invariant)
    if len(prompt) > 1:
        target.forward(torch.tensor([prompt[:-1]], device=dev), cache=tc, last_only=True)
        draft.forward(torch.tensor([prompt[:-1]], device=dev), cache=dc, last_only=True)
        res.n_forward_target += 1; res.n_forward_draft += 1
    _sync(dev); res.prefill_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    last = prompt[-1]
    done = False
    first_token_time = None
    while not done and len(res.tokens) < max_new:
        kk = min(k, max_new - len(res.tokens))     # don't draft past the budget
        # 1) draft k tokens
        drafts, cur = [], last
        for _ in range(kk):
            lg = draft.forward(torch.tensor([[cur]], device=dev), cache=dc, last_only=True)
            cur = _pick(lg[0, -1], eos_ids, len(res.tokens) + len(drafts) >= min_new)
            drafts.append(cur)
            res.n_forward_draft += 1
            if cur in eos_ids:
                break
        kk = len(drafts)
        # 2) one target pass over [last, d1..dk]
        inp = torch.tensor([[last] + drafts], device=dev)
        lg = target.forward(inp, cache=tc, last_only=False)[0]               # (kk+1, V)
        res.n_forward_target += 1
        greedy = []
        for i in range(kk + 1):
            greedy.append(_pick(lg[i], eos_ids, len(res.tokens) + i >= min_new))
        # 3) accept the longest matching prefix, then take the target's token
        n = 0
        while n < kk and drafts[n] == greedy[n]:
            n += 1
        new = drafts[:n] + [greedy[n]]
        proposed += kk; accepted += n; rounds += 1
        accepted_hist.append(n)
        # 4) cache rollback / sync.  Target now holds last,d1..dk; keep last,d1..dn.
        base_t = tc.length - (kk + 1)
        tc.truncate(base_t + n + 1)
        if n == kk:
            # all accepted: draft never consumed d_k -> feed it so the invariant holds
            if drafts[-1] not in eos_ids:
                draft.forward(torch.tensor([[drafts[-1]]], device=dev), cache=dc, last_only=True)
                res.n_forward_draft += 1
        else:
            base_d = dc.length - kk if kk else dc.length   # dc advanced kk times (last,d1..d_{kk-1}) -> kk entries
            dc.truncate(base_d + n + 1)
        for tkn in new:
            res.tokens.append(tkn)
            if first_token_time is None:
                first_token_time = time.perf_counter() - t0
            if tkn in eos_ids or len(res.tokens) >= max_new:
                done = True
                break
        last = res.tokens[-1]
    _sync(dev)
    res.decode_s = time.perf_counter() - t1
    res.stats.update({
        "k": k, "rounds": rounds, "proposed": proposed, "accepted": accepted,
        "acceptance_rate": accepted / proposed if proposed else float("nan"),
        "mean_accepted_per_round": accepted / rounds if rounds else float("nan"),
        "tokens_per_target_pass": len(res.tokens) / max(res.n_forward_target, 1),
        "accepted_hist": accepted_hist,
        "ttft_s": first_token_time,
    })
    return res
