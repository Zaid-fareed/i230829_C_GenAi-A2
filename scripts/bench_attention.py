"""Task 4 benchmark: reference vs tiled(online softmax) vs PyTorch SDPA attention.

Shapes follow the 360M model (15 query heads, 5 KV heads, head_dim 64).  Reports max / mean
absolute error against the reference, median runtime, and sampled peak-RSS growth (each point in
its own subprocess).  Both causal prefill (Tq = Tk) and decode-style (Tq = 1 with a long cache) are timed.
"""
import argparse
import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def worker(a):
    import torch
    torch.set_num_threads(a.threads)
    from attention import BACKENDS
    from memprobe import PeakRSS
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    B, H, Hkv, D = 1, 15, 5, 64
    Tq = a.T if a.kind == "prefill" else 1
    Tk = a.T
    off = 0 if a.kind == "prefill" else a.T - 1
    q = torch.randn(B, H, Tq, D, device=dev); k = torch.randn(B, Hkv, Tk, D, device=dev); v = torch.randn(B, Hkv, Tk, D, device=dev)
    fn = BACKENDS[a.impl]
    kw = dict(block_q=a.block, block_k=a.block) if a.impl == "tiled" else {}
    ref = BACKENDS["reference"](q, k, v, q_offset=off)
    fn(q, k, v, q_offset=off, **kw)                                    # warm-up
    times = []
    with PeakRSS() as pk:
        for _ in range(a.repeats):
            torch.cuda.synchronize() if dev == "cuda" else None; t0 = time.perf_counter(); out = fn(q, k, v, q_offset=off, **kw); torch.cuda.synchronize() if dev == "cuda" else None; times.append(time.perf_counter() - t0)
    err = (out - ref).abs()
    score_mb = B * H * Tq * Tk * 4 / 2**20
    print("RESULT" + json.dumps({"impl": a.impl, "kind": a.kind, "T": a.T, "block": a.block if a.impl == "tiled" else None,
                                 "median_ms": st.median(times) * 1e3, "max_abs_err": err.max().item(),
                                 "mean_abs_err": err.mean().item(), "peak_delta_mb": pk.delta_mb,
                                 "full_score_matrix_mb": score_mb}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--impl"); ap.add_argument("--kind"); ap.add_argument("--T", type=int)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--lengths", type=int, nargs="*", default=[128, 256, 512, 1024, 2048])
    ap.add_argument("--out", default="results/metrics/bench_attention.json")
    a = ap.parse_args()
    if a.worker:
        return worker(a)
    res = []
    plan = [(impl, kind, T, blk) for kind in ("prefill", "decode") for T in a.lengths
            for impl, blk in (("reference", None), ("sdpa", None), ("tiled", 64))]
    plan += [("tiled", "prefill", 1024, b) for b in (16, 32, 128, 256)]          # block-size sweep
    for impl, kind, T, blk in plan:
        cmd = [sys.executable, __file__, "--worker", "--impl", impl, "--kind", kind, "--T", str(T),
               "--repeats", str(a.repeats), "--threads", str(a.threads)] + (["--block", str(blk)] if blk else [])
        p = subprocess.run(cmd, capture_output=True, text=True)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT")]
        if not line:
            print(p.stderr[-1500:]); raise SystemExit("worker failed")
        r = json.loads(line[0][6:]); res.append(r)
        print(f"{kind:8s} T={T:5d} {impl:9s} blk={r['block']} {r['median_ms']:9.2f} ms  maxerr={r['max_abs_err']:.2e} "
              f"peakΔ={r['peak_delta_mb']:.0f}MB", flush=True)
    Path(ROOT / a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
