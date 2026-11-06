"""A from-scratch Llama-architecture decoder that reads Hugging Face weights.

Why not just call ``model.generate``?  Tasks 3-5 require us to control the KV
cache, position ids, masks and the attention kernel ourselves.  This module
re-implements the forward pass of ``LlamaForCausalLM`` (RMSNorm, rotary
embeddings, grouped-query attention, SwiGLU MLP) with

* an explicit, growable ``KVCache`` that can be truncated (needed to roll back
  rejected speculative tokens),
* explicit position handling (positions = cache length + arange(T)),
* a pluggable attention back-end (reference / tiled online-softmax / SDPA).

Correctness against the Hugging Face implementation is checked in
``tests/test_minillama.py``.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from attention import BACKENDS


@dataclass
class LlamaCfg:
    hidden: int
    inter: int
    layers: int
    heads: int
    kv_heads: int
    vocab: int
    eps: float
    theta: float
    max_pos: int
    tie: bool

    @property
    def head_dim(self):
        return self.hidden // self.heads

    @classmethod
    def from_json(cls, path):
        c = json.loads(Path(path).read_text())
        return cls(c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"], c["num_attention_heads"],
                   c.get("num_key_value_heads", c["num_attention_heads"]), c["vocab_size"],
                   c["rms_norm_eps"], c.get("rope_theta", 10000.0), c["max_position_embeddings"],
                   c.get("tie_word_embeddings", False))


class KVCache:
    """Per-layer key/value storage with amortised growth and rollback.

    Storage is a pre-allocated tensor of shape (B, Hkv, capacity, D) per layer.
    ``length`` is the number of valid positions.  New keys/values are written at
    [length, length+T); the buffer doubles when full (cache growth = O(1)
    amortised copies instead of one ``torch.cat`` per token).  ``truncate(n)``
    simply lowers ``length`` – stale entries beyond it are overwritten later.
    """

    def __init__(self, cfg: LlamaCfg, batch=1, capacity=256, dtype=torch.float32, device="cpu"):
        self.cfg, self.batch, self.dtype, self.device = cfg, batch, dtype, device
        self.length = 0
        self.capacity = capacity
        shape = (batch, cfg.kv_heads, capacity, cfg.head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(cfg.layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(cfg.layers)]
        self.n_grow = 0

    def _ensure(self, need: int):
        if need <= self.capacity:
            return
        new_cap = self.capacity
        while new_cap < need:
            new_cap *= 2
        for i in range(self.cfg.layers):
            for buf in (self.k, self.v):
                nb = torch.zeros(buf[i].shape[:2] + (new_cap, buf[i].shape[3]), dtype=self.dtype, device=self.device)
                nb[:, :, : self.length] = buf[i][:, :, : self.length]
                buf[i] = nb
        self.capacity = new_cap
        self.n_grow += 1

    def write(self, layer: int, k_new, v_new):
        """Store the new K/V of ``layer`` at [length, length+T) and return views of all valid K/V."""
        t = k_new.shape[2]
        self._ensure(self.length + t)
        self.k[layer][:, :, self.length: self.length + t] = k_new
        self.v[layer][:, :, self.length: self.length + t] = v_new
        return self.k[layer][:, :, : self.length + t], self.v[layer][:, :, : self.length + t]

    def advance(self, t: int):
        self.length += t

    def truncate(self, n: int):
        assert 0 <= n <= self.length
        self.length = n

    def nbytes_used(self):
        per = self.batch * self.cfg.kv_heads * self.length * self.cfg.head_dim * torch.tensor([], dtype=self.dtype).element_size()
        return 2 * self.cfg.layers * per

    def nbytes_allocated(self):
        return sum(t.numel() * t.element_size() for t in self.k + self.v)


def rms_norm(x, w, eps):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * w.float()).to(x.dtype)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class MiniLlama:
    def __init__(self, model_dir, dtype=torch.float32, attn="sdpa", device="cpu"):
        model_dir = Path(model_dir)
        self.cfg = LlamaCfg.from_json(model_dir / "config.json")
        self.dtype, self.device = dtype, device
        self.attn_name = attn
        self.attn = BACKENDS[attn]
        sd = {}
        for f in sorted(model_dir.glob("*.safetensors")):
            sd.update(load_file(str(f)))
        self.w = {k: v.to(dtype=dtype, device=device) for k, v in sd.items()}
        if self.cfg.tie or "lm_head.weight" not in self.w:
            self.w["lm_head.weight"] = self.w["model.embed_tokens.weight"]
        inv = 1.0 / (self.cfg.theta ** (torch.arange(0, self.cfg.head_dim, 2, dtype=torch.float32) / self.cfg.head_dim))
        self.inv_freq = inv.to(device)
        self.n_params = sum(t.numel() for k, t in self.w.items() if k != "lm_head.weight") + (
            0 if self.cfg.tie else self.w["lm_head.weight"].numel())

    def merge_lora(self, adapter_dir):
        """Fold a PEFT LoRA adapter into the weights:  W <- W + (alpha / r) * B @ A.

        Done by hand (no peft needed at inference time) so the adapter can be shipped
        as a ~35 MB file and applied to the downloaded base model.
        """
        adapter_dir = Path(adapter_dir)
        acfg = json.loads((adapter_dir / "adapter_config.json").read_text())
        scale = acfg["lora_alpha"] / acfg["r"]
        ad = load_file(str(adapter_dir / "adapter_model.safetensors"))
        n = 0
        for key, a in ad.items():
            if ".lora_A." not in key:
                continue
            b = ad[key.replace(".lora_A.", ".lora_B.")]
            target = key.replace("base_model.model.", "", 1).replace(".lora_A.weight", ".weight")
            delta = (b.float() @ a.float()) * scale
            self.w[target] = (self.w[target].float() + delta.to(self.device)).to(self.dtype)
            n += 1
        if self.cfg.tie:                      # lm_head shares the embedding tensor; untouched by LoRA here
            self.w["lm_head.weight"] = self.w["model.embed_tokens.weight"]
        return n

    def set_attention(self, name):
        self.attn_name, self.attn = name, BACKENDS[name]

    def new_cache(self, batch=1, capacity=256):
        return KVCache(self.cfg, batch, capacity, self.dtype, self.device)

    # -- rotary embedding with explicit absolute positions --------------------------------
    def _rope(self, pos):
        freqs = pos.float()[:, None] * self.inv_freq[None, :]                # (T, D/2)
        emb = torch.cat((freqs, freqs), dim=-1)                              # (T, D)
        return emb.cos()[None, None], emb.sin()[None, None]                  # (1,1,T,D)

    @torch.no_grad()
    def forward(self, input_ids, cache: KVCache | None = None, last_only=False, key_mask=None):
        """Run the decoder on ``input_ids`` (B, T).

        With a cache the new tokens get absolute positions cache.length .. cache.length+T-1,
        their K/V are appended, and attention sees [cached ; new] keys with the causal rule
        "query at absolute position p sees keys <= p".  Without a cache the whole sequence is
        recomputed from position 0 (the reference path).
        Returns logits (B, T, V) or (B, 1, V) if ``last_only``.
        """
        cfg, w = self.cfg, self.w
        B, T = input_ids.shape
        offset = cache.length if cache is not None else 0
        pos = torch.arange(offset, offset + T, device=input_ids.device)
        cos, sin = self._rope(pos)
        h = F.embedding(input_ids, w["model.embed_tokens.weight"])
        H, Hkv, D = cfg.heads, cfg.kv_heads, cfg.head_dim
        for i in range(cfg.layers):
            p = f"model.layers.{i}."
            x = rms_norm(h, w[p + "input_layernorm.weight"], cfg.eps)
            q = F.linear(x, w[p + "self_attn.q_proj.weight"]).view(B, T, H, D).transpose(1, 2)
            k = F.linear(x, w[p + "self_attn.k_proj.weight"]).view(B, T, Hkv, D).transpose(1, 2)
            v = F.linear(x, w[p + "self_attn.v_proj.weight"]).view(B, T, Hkv, D).transpose(1, 2)
            q = q * cos + rotate_half(q) * sin       # RoPE is applied BEFORE caching: cached keys already
            k = k * cos + rotate_half(k) * sin       # carry their absolute-position rotation.
            if cache is not None:
                k_all, v_all = cache.write(i, k, v)
            else:
                k_all, v_all = k, v
            a = self.attn(q, k_all, v_all, q_offset=offset, causal=True, key_mask=key_mask)
            a = a.transpose(1, 2).reshape(B, T, H * D)
            h = h + F.linear(a, w[p + "self_attn.o_proj.weight"])
            x = rms_norm(h, w[p + "post_attention_layernorm.weight"], cfg.eps)
            g = F.silu(F.linear(x, w[p + "mlp.gate_proj.weight"])) * F.linear(x, w[p + "mlp.up_proj.weight"])
            h = h + F.linear(g, w[p + "mlp.down_proj.weight"])
        if cache is not None:
            cache.advance(T)
        h = rms_norm(h, w["model.norm.weight"], cfg.eps)
        if last_only:
            h = h[:, -1:]
        return F.linear(h, w["lm_head.weight"])
