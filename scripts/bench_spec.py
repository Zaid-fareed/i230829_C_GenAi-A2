"""Task 5 benchmark: greedy target decoding vs speculative decoding.

For every prompt (a held-out TEST criterion prompt; only speed/equality is measured, no tuning) we run
  1. plain cached greedy target decoding            (reference)
  2. our speculative decoder for k in K_LIST         with the fine-tuned draft and with the un-tuned draft
and check token equality.  Results: acceptance statistics, latency, tokens/s, speed-up, peak RSS.
A forced-length setting (EOS suppressed, 64 new tokens) is also run to show behaviour on longer outputs.
Finally, Hugging Face assisted generation is used as an independent token-equality reference.
"""
import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--k_list", type=int, nargs="*", default=[2, 4, 6, 8])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--hf_check", type=int, default=6)
    ap.add_argument("--out", default="results/metrics/bench_spec.json")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(a.threads)
    import data as D
    from memprobe import PeakRSS
    from pipeline import Engine, DRAFT_BASE, DRAFT_ADAPTER, TARGET_BASE, TARGET_ADAPTER, DEVICE
    from minillama import MiniLlama

    eng = Engine()
    untuned = MiniLlama(DRAFT_BASE, device=DEVICE)
    df = D.load_split("test").sample(n=a.n, random_state=0)
    prompts = [eng.encode(r["note"], r["criterion_type"], r["criterion_text"]) for _, r in df.iterrows()]
    eng.generate(prompts[0], "kv_cache", max_new=4)                                      # warm-up

    results = {"n_prompts": len(prompts), "settings": {}}
    for setting, (max_new, min_new) in {"natural_output": (48, 0), "forced_64_tokens": (64, 64)}.items():
        ref_runs = []
        with PeakRSS() as pk_ref:
            for ids in prompts:
                ref_runs.append(eng.generate(ids, "kv_cache", max_new, min_new))
        base = {"latency_s": st.mean(r.total_s for r in ref_runs),
                "tok_per_s": sum(len(r.tokens) for r in ref_runs) / sum(r.total_s for r in ref_runs),
                "peak_delta_mb": pk_ref.delta_mb, "mean_new_tokens": st.mean(len(r.tokens) for r in ref_runs)}
        entry = {"greedy_target": base, "speculative": []}
        for draft_name, draft_model in (("finetuned_draft", eng.draft), ("untuned_draft", untuned)):
            for k in a.k_list:
                eng.draft_backup = eng.draft; eng.draft = draft_model
                runs, equal = [], 0
                with PeakRSS() as pk:
                    for ids, ref in zip(prompts, ref_runs):
                        r = eng.generate(ids, "speculative", max_new, min_new, k=k)
                        runs.append(r); equal += int(r.tokens == ref.tokens)
                eng.draft = eng.draft_backup
                prop = sum(r.stats["proposed"] for r in runs); acc = sum(r.stats["accepted"] for r in runs)
                lat = st.mean(r.total_s for r in runs)
                row = {"draft": draft_name, "k": k, "equal_to_greedy": f"{equal}/{len(runs)}",
                       "acceptance_rate": acc / prop if prop else None,
                       "mean_accepted_per_round": acc / max(sum(r.stats["rounds"] for r in runs), 1),
                       "target_passes_per_token": sum(r.n_forward_target for r in runs) / sum(len(r.tokens) for r in runs),
                       "latency_s": lat, "tok_per_s": sum(len(r.tokens) for r in runs) / sum(r.total_s for r in runs),
                       "speedup_vs_greedy": base["latency_s"] / lat, "peak_delta_mb": pk.delta_mb}
                entry["speculative"].append(row)
                print(setting, row, flush=True)
        results["settings"][setting] = entry

    # independent reference: Hugging Face assisted generation must match our greedy tokens too
    from transformers import AutoModelForCausalLM
    hf_t = AutoModelForCausalLM.from_pretrained(TARGET_BASE, dtype=torch.float32).eval()
    hf_d = AutoModelForCausalLM.from_pretrained(DRAFT_BASE, dtype=torch.float32).eval()
    from peft import PeftModel
    hf_t = PeftModel.from_pretrained(hf_t, TARGET_ADAPTER).merge_and_unload().eval()
    hf_d = PeftModel.from_pretrained(hf_d, DRAFT_ADAPTER).merge_and_unload().eval()
    ok = 0
    for ids in prompts[: a.hf_check]:
        inp = torch.tensor([ids])
        with torch.no_grad():
            out = hf_t.generate(inp, assistant_model=hf_d, max_new_tokens=48, do_sample=False,
                                eos_token_id=list(eng.eos), pad_token_id=eng.tok.eos_token_id)
        hf_tokens = out[0, len(ids):].tolist()
        mine = eng.generate(ids, "speculative", 48, 0).tokens
        ref = eng.generate(ids, "kv_cache", 48, 0).tokens
        trim = lambda t: t[: t.index(next(e for e in eng.eos if e in t)) + 1] if any(e in t for e in eng.eos) else t
        ok += int(trim(hf_tokens) == mine == ref)
    results["hf_assisted_generation_check"] = f"{ok}/{a.hf_check} prompts: HF assisted == ours == cached greedy"
    print(results["hf_assisted_generation_check"])
    (ROOT / a.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
