"""LoRA fine-tuning of a causal LM on the training split (CPU friendly).

Loss is computed only on the answer tokens (the JSON string and the end-of-turn
token).  After every epoch the script logs validation loss and runs greedy
generation on the validation split to compute accuracy / macro-F1, and saves the
adapter.  The test split is never read here.

Usage: python scripts/train.py --config configs/finetune_360m.json
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data as D  # noqa: E402
import metrics as M  # noqa: E402


def encode_example(tok, row, max_len):
    sents = row["note"]
    msgs = D.build_messages(sents, row["criterion_type"], row["criterion_text"])
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    answer = D.encode_target(row["expert_eligibility"], row["expert_sentences"]) + "<|im_end|>"
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(answer, add_special_tokens=False)["input_ids"]
    truncated = False
    if len(p_ids) + len(a_ids) > max_len:
        # keep the head of the prompt (system + note start) is NOT done: we cut from the note middle
        # only if absolutely necessary; the event is counted and reported.
        truncated = True
        p_ids = p_ids[: max_len - len(a_ids)]
    return p_ids, a_ids, truncated


def collate(batch, pad_id):
    L = max(len(p) + len(a) for p, a in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (p, a) in enumerate(batch):
        s = p + a
        ids[i, : len(s)] = torch.tensor(s)
        lab[i, len(p): len(s)] = torch.tensor(a)
        att[i, : len(s)] = 1
    return ids, att, lab


@torch.no_grad()
def val_loss(model, data, pad_id, bs=8, dev="cpu"):
    model.eval()
    tot, cnt = 0.0, 0
    for i in range(0, len(data), bs):
        ids, att, lab = [t.to(dev) for t in collate(data[i:i + bs], pad_id)]
        out = model(input_ids=ids, attention_mask=att, labels=lab)
        n = (lab[:, 1:] != -100).sum().item()
        tot += out.loss.item() * n
        cnt += n
    return tot / cnt


@torch.no_grad()
def generate_eval(model, tok, df, bs=8, max_new=48, dev="cpu"):
    model.eval()
    tok.padding_side = "left"
    rows = []
    prompts = []
    for _, r in df.iterrows():
        msgs = D.build_messages(r["note"], r["criterion_type"], r["criterion_text"])
        prompts.append(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))
    order = np.argsort([len(p) for p in prompts])
    preds = {}
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        enc = tok([prompts[j] for j in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for j, t in zip(idx, texts):
            preds[j] = t
    for j, (_, r) in enumerate(df.iterrows()):
        lab, ev, ok = D.parse_output(preds[j], r["criterion_type"])
        rows.append({"criterion_type": r["criterion_type"], "gold_label": r["expert_eligibility"],
                     "pred_label": lab, "gold_ev": r["expert_sentences"], "pred_ev": ev})
    return M.full_report(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_steps", type=int, default=None, help="debug: stop early")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())

    torch.set_num_threads(cfg.get("threads", 4))
    if cfg.get("seed_override"):
        pass
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])

    base = ROOT / cfg["base_model"]
    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / cfg["log_csv"]
    log_path.parent.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(base)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = cfg.get("amp", "none") if dev == "cuda" else "none"      # "fp16" (T4/P100) or "bf16" (Ampere+)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).to(dev)
    print("device", dev, "amp", amp, flush=True)
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp == "fp16"))
    lcfg = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                      target_modules=cfg["target_modules"], bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    train_df, val_df = D.load_split("train"), D.load_split("validation")
    enc_train = [encode_example(tok, r, cfg["max_len"]) for _, r in train_df.iterrows()]
    enc_val = [encode_example(tok, r, cfg["max_len"]) for _, r in val_df.iterrows()]
    n_trunc = sum(t for *_, t in enc_train) + sum(t for *_, t in enc_val)
    lens = [len(p) + len(a) for p, a, _ in enc_train + enc_val]
    stats = {"n_train": len(enc_train), "n_val": len(enc_val), "max_len_setting": cfg["max_len"],
             "max_tokens_seen": int(max(lens)), "mean_tokens": float(np.mean(lens)),
             "n_truncated": int(n_trunc)}
    print("data stats", stats, flush=True)
    (out_dir / "data_stats.json").write_text(json.dumps(stats, indent=2))
    train_pairs = [(p, a) for p, a, _ in enc_train]
    val_pairs = [(p, a) for p, a, _ in enc_val]

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    bs, accum, epochs = cfg["batch_size"], cfg["grad_accum"], cfg["epochs"]
    steps_per_epoch = math.ceil(len(train_pairs) / (bs * accum))
    total = steps_per_epoch * epochs
    warm = max(1, int(cfg["warmup_ratio"] * total))

    def lr_at(s):
        if s < warm:
            return cfg["lr"] * (s + 1) / warm
        prog = (s - warm) / max(1, total - warm)
        return cfg["lr"] * 0.5 * (1 + math.cos(math.pi * prog))

    with open(log_path, "w") as f:
        f.write("step,epoch,train_loss,lr,elapsed_s\n")
    summary_path = out_dir / "epoch_summary.json"
    history = []
    step, t0 = 0, time.time()
    rng = random.Random(cfg["seed"])
    for ep in range(1, epochs + 1):
        model.train()
        order = list(range(len(train_pairs)))
        rng.shuffle(order)
        # bucket by length inside windows of 8 micro-batches to cut padding waste
        win = bs * accum
        micro = []
        for i in range(0, len(order), win):
            chunk = sorted(order[i:i + win], key=lambda j: len(train_pairs[j][0]) + len(train_pairs[j][1]))
            micro.append(chunk)
        for chunk in micro:
            opt.zero_grad()
            n_tok_total = sum(len(train_pairs[j][1]) for j in chunk)
            loss_acc = 0.0
            for k in range(0, len(chunk), bs):
                sub = [train_pairs[j] for j in chunk[k:k + bs]]
                ids, att, lab = [t.to(dev) for t in collate(sub, tok.pad_token_id)]
                with torch.autocast(dev, dtype=amp_dtype, enabled=amp_dtype is not None):
                    logits = model(input_ids=ids, attention_mask=att).logits
                logits = logits.float()
                loss_sum = torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)), lab[:, 1:].reshape(-1),
                    ignore_index=-100, reduction="sum")
                scaler.scale(loss_sum / n_tok_total).backward()
                loss_acc += loss_sum.item()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            scaler.step(opt); scaler.update()
            step += 1
            with open(log_path, "a") as f:
                f.write(f"{step},{ep},{loss_acc / n_tok_total:.5f},{lr_at(step - 1):.3e},{time.time() - t0:.1f}\n")
            print(f"ep {ep} step {step}/{total} loss {loss_acc / n_tok_total:.4f} "
                  f"elapsed {time.time() - t0:.0f}s", flush=True)
            if args.max_steps and step >= args.max_steps:
                return
        vl = val_loss(model, val_pairs, tok.pad_token_id, dev=dev)
        rep = generate_eval(model, tok, val_df, dev=dev)
        ov = rep["overall"]
        entry = {"epoch": ep, "step": step, "val_loss": vl,
                 "val_accuracy": ov["classification"]["accuracy"],
                 "val_macro_f1": ov["classification"]["macro_f1"],
                 "val_evidence_micro_f1": ov["evidence"]["micro"]["f1"],
                 "elapsed_s": time.time() - t0}
        history.append(entry)
        print("EPOCH", entry, flush=True)
        model.save_pretrained(out_dir / f"epoch{ep}")
        (out_dir / f"epoch{ep}" / "val_report.json").write_text(json.dumps(rep, indent=2))
        summary_path.write_text(json.dumps(history, indent=2))
    best = max(history, key=lambda h: (h["val_macro_f1"], -h["val_loss"]))
    (out_dir / "best.json").write_text(json.dumps({"best_epoch": best["epoch"], "selection": "max val macro-F1, ties by lower val loss", **best}, indent=2))
    print("BEST", best, flush=True)


if __name__ == "__main__":
    main()
