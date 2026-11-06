"""Task 1 evidence: measure two open-weight decoder candidates on identical inputs.

For each candidate we record parameter count, fp32 weight memory, tokenizer sizes, token counts for the task
prompts, one LoRA training-step time (batch of 4 real training examples, forward+backward) and zero-shot
validation behaviour (well-formed output rate).  Only TRAIN/VALIDATION data are touched.
"""
import json
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import data as D  # noqa: E402
from train import collate, encode_example  # noqa: E402

CANDS = {"SmolLM2-360M-Instruct": ROOT / "checkpoints/base/SmolLM2-360M-Instruct",
         "Qwen2.5-0.5B-Instruct": ROOT / "checkpoints/base/Qwen2.5-0.5B-Instruct",
         "SmolLM2-135M-Instruct": ROOT / "checkpoints/base/SmolLM2-135M-Instruct"}
torch.set_num_threads(4)
train = D.load_split("train").head(16)
val = D.load_split("validation").sample(n=30, random_state=0)
out = {}
for name, path in CANDS.items():
    if not path.exists():
        print("missing", name); continue
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    cfg = model.config
    n = sum(p.numel() for p in model.parameters())
    enc = [encode_example(tok, r, 1024) for _, r in train.iterrows()]
    lens = [len(p) + len(a) for p, a, _ in enc]
    # one LoRA step
    pm = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
    ids, att, lab = collate([(p, a) for p, a, _ in enc[:4]], tok.pad_token_id or 0)
    pm.train(); t0 = time.perf_counter()
    loss = pm(input_ids=ids, attention_mask=att, labels=lab).loss; loss.backward()
    step_s = time.perf_counter() - t0
    # zero-shot
    pm.eval(); ok = 0; t1 = time.perf_counter()
    tok.padding_side = "left"
    for _, r in val.iterrows():
        text = tok.apply_chat_template(D.build_messages(r["note"], r["criterion_type"], r["criterion_text"]), add_generation_prompt=True, tokenize=False)
        x = tok(text, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            g = pm.generate(**x, max_new_tokens=48, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        txt = tok.decode(g[0, x["input_ids"].shape[1]:], skip_special_tokens=True)
        ok += int(D.parse_output(txt, r["criterion_type"])[2])
    out[name] = {"parameters_M": n / 1e6, "fp32_weights_MB": n * 4 / 2**20, "layers": cfg.num_hidden_layers,
                 "hidden": cfg.hidden_size, "heads": cfg.num_attention_heads, "kv_heads": cfg.num_key_value_heads,
                 "vocab": cfg.vocab_size, "max_position_embeddings": cfg.max_position_embeddings,
                 "mean_train_example_tokens": sum(lens) / len(lens),
                 "lora_step_batch4_seconds": step_s, "zero_shot_well_formed_json": f"{ok}/{len(val)}",
                 "zero_shot_seconds_per_example": (time.perf_counter() - t1) / len(val)}
    print(name, out[name], flush=True)
    del model, pm
(ROOT / "results" / "metrics" / "candidate_compare.json").write_text(json.dumps(out, indent=2))
