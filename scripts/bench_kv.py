"""Task 3 benchmark: uncached reference vs explicit KV-cache decoding.

Each (input_len, output_len, mode) point runs in a fresh subprocess (clean allocator) with
1 warm-up + N timed repeats; we report medians.  EOS is suppressed (min_new = out_len) so that
both paths generate exactly the same number of tokens, and greedy outputs must be identical.

  python scripts/bench_kv.py                      # full grid
  python scripts/bench_kv.py --worker ...         # (internal)
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

GRID_IN = [64, 256, 512]
GRID_OUT = [16, 64, 128]


def make_prompt(eng, n_in):
    import data as D
    df = D.load_split("test")
    text = ""
    ids = []
    for _, r in df.iterrows():
        ids += eng.encode(r["note"], r["criterion_type"], r["criterion_text"])
        if len(ids) >= n_in:
            break
    return ids[:n_in]


def worker(args):
    import torch
    torch.set_num_threads(args.threads)
    from memprobe import PeakRSS
    from pipeline import Engine
    eng = Engine(load_draft=False)
    ids = make_prompt(eng, args.n_in)
    mode = args.mode
    eng.generate(ids[:32], mode, max_new=4, min_new=4)           # warm-up (kernels, allocator)
    runs, toks = [], None
    with PeakRSS() as pk:
        for _ in range(args.repeats):
            r = eng.generate(ids, mode, max_new=args.n_out, min_new=args.n_out)
            runs.append(r)
            toks = r.tokens if toks is None else toks
            assert r.tokens == toks, "non-deterministic output between repeats"
    kv_bytes = r.stats.get("kv_bytes_used", 0)
    out = {
        "mode": mode, "n_in": args.n_in, "n_out": args.n_out, "repeats": args.repeats,
        "prefill_s": st.median(x.prefill_s for x in runs),
        "decode_s": st.median(x.decode_s for x in runs),
        "total_s": st.median(x.total_s for x in runs),
        "tok_per_s": st.median(x.tok_per_s for x in runs),
        "decode_tok_per_s": st.median(x.decode_tok_per_s for x in runs),
        "peak_rss_mb": pk.peak_mb, "peak_delta_mb": pk.delta_mb,
        "kv_cache_mb": kv_bytes / 2**20,
        "tokens": toks,
    }
    print("RESULT" + json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--mode"); ap.add_argument("--n_in", type=int); ap.add_argument("--n_out", type=int)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--grid_in", type=int, nargs="*", default=GRID_IN)
    ap.add_argument("--grid_out", type=int, nargs="*", default=GRID_OUT)
    ap.add_argument("--modes", nargs="*", default=["uncached", "kv_cache", "kv_cache_tiled"])
    ap.add_argument("--out", default="results/metrics/bench_kv.json")
    args = ap.parse_args()
    if args.worker:
        return worker(args)
    results = []
    for n_in in args.grid_in:
        for n_out in args.grid_out:
            row = {}
            for mode in args.modes:
                cmd = [sys.executable, __file__, "--worker", "--mode", mode, "--n_in", str(n_in),
                       "--n_out", str(n_out), "--repeats", str(args.repeats), "--threads", str(args.threads)]
                t0 = time.time()
                p = subprocess.run(cmd, capture_output=True, text=True)
                line = [l for l in p.stdout.splitlines() if l.startswith("RESULT")]
                if not line:
                    print(p.stderr[-2000:]); raise SystemExit("worker failed")
                r = json.loads(line[0][6:]); row[mode] = r
                print(f"in={n_in:4d} out={n_out:4d} {mode:15s} prefill={r['prefill_s']:.3f}s decode={r['decode_s']:.3f}s "
                      f"tok/s={r['tok_per_s']:.2f} peakΔ={r['peak_delta_mb']:.0f}MB ({time.time() - t0:.0f}s)", flush=True)
            ref = row[args.modes[0]]["tokens"]
            for mode in args.modes:
                row[mode]["tokens_match_reference"] = row[mode]["tokens"] == ref
                row[mode]["n_tokens"] = len(row[mode].pop("tokens"))
                results.append(row[mode])
    Path(ROOT / args.out).write_text(json.dumps(results, indent=2))
    print("saved", args.out)


if __name__ == "__main__":
    main()
