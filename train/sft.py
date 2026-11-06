"""Instruction fine-tuning (SFT) of a base student on one data set (a teacher's replies, or the human replies).

Runs that are compared must see the same prompts: --align lists every data set in the comparison, and only
prompts that all of them answered with a finished reply (finish == "stop") are used, in the order of
$OUT_DIR/data/prompts/train.jsonl, first --n of them. Hyperparameters are the same for every data set.

Format: configs/student_chat.jinja, loss on reply tokens only (prompt/completion data). The chat template
is saved with every checkpoint, so prefs_eval.py --format chat runs the student in its training format.

Usage:
  python train/sft.py --student olmo2_1b --data gemma3_27b --align gemma3_27b,olmo2_32b,human --n 5000
Writes $OUT_DIR/ckpt/<student>/<data>_n<n>_s<seed>/checkpoint-<step> at --save-fracs of training,
plus train_info.json (prompts used, reply tokens, loss curve).
"""
import argparse
import json
import math
import os
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.environ.get("OUT_DIR", ".")
DATA = os.path.join(OUT, "data")


def load(name):
    return {r["id"]: r for r in map(json.loads, open(os.path.join(DATA, name, "train.jsonl")))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", required=True, help="name in configs/models.yaml students, or an HF id")
    ap.add_argument("--data", required=True, help="data set under $OUT_DIR/data (teacher name or human)")
    ap.add_argument("--align", help="comma-separated data sets that must all have a finished reply")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=8, help="per-device batch")
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation (effective batch = batch*accum)")
    ap.add_argument("--max-length", type=int, default=4096, help="prompt + reply tokens (teacher replies go up to 3072)")
    ap.add_argument("--save-fracs", default="0.25,0.5,1.0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chat-template", default=os.path.join(ROOT, "configs", "student_chat.jinja"))
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    students = yaml.safe_load(open(os.path.join(ROOT, "configs", "models.yaml")))["students"]
    model_id = students[args.student]["id"] if args.student in students else args.student
    sets = {d: load(d) for d in dict.fromkeys([args.data] + (args.align.split(",") if args.align else []))}
    order = [json.loads(line)["id"] for line in open(os.path.join(DATA, "prompts", "train.jsonl"))]
    ids = [i for i in order if all(i in s and s[i]["finish"] == "stop" and s[i]["reply"] for s in sets.values())]
    if len(ids) < args.n:
        raise SystemExit(f"only {len(ids)} prompts have a finished reply in every set of {list(sets)}")
    ids = ids[:args.n]
    rows = [sets[args.data][i] for i in ids]

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.chat_template = open(args.chat_template).read()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = Dataset.from_list([{"prompt": [{"role": "user", "content": r["prompt"]}],
                             "completion": [{"role": "assistant", "content": r["reply"]}]} for r in rows])

    run = f"{args.data}_n{args.n}_s{args.seed}"
    out = os.path.join(OUT, "ckpt", args.student, run)
    os.makedirs(out, exist_ok=True)
    steps = math.ceil(math.ceil(args.n / args.batch) / args.accum * args.epochs)
    save_at = sorted({max(1, round(f * steps)) for f in map(float, args.save_fracs.split(","))})

    class SaveAt(TrainerCallback):
        def on_step_end(self, a, state, control, **kw):
            if state.global_step in save_at:
                control.should_save = True

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)  # fp32 master weights, bf16 autocast
    cfg = SFTConfig(output_dir=out, run_name=f"{args.student}/{run}", num_train_epochs=args.epochs,
                    learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=0.03, weight_decay=0.0,
                    per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.accum,
                    max_length=args.max_length, completion_only_loss=True, packing=False, bf16=True,
                    gradient_checkpointing=True, logging_steps=10, save_strategy="no", save_only_model=True,
                    seed=args.seed, data_seed=args.seed, report_to=["wandb"] if args.wandb else [],
                    dataloader_num_workers=2)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, callbacks=[SaveAt()])

    # Check the loss mask: only the reply (and its end token) is trained on. TRL >= 1.x folds the completion
    # mask into `labels` (-100 = not trained) and drops the completion_mask column.
    ex = trainer.train_dataset[0]
    if "labels" not in ex:
        raise SystemExit(f"no labels column (columns: {trainer.train_dataset.column_names}); loss mask unknown")
    print("example input:", repr(tok.decode(ex["input_ids"])[-300:]))
    print("trained part: ", repr(tok.decode([t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100])[-200:]))
    print("untrained end:", repr(tok.decode([t for t, l in zip(ex["input_ids"], ex["labels"]) if l == -100])[-200:]))
    lens = [sum(l != -100 for l in labels) for labels in trainer.train_dataset["labels"]]
    if any(n == len(ids) for n, ids in zip(lens, trainer.train_dataset["input_ids"])):
        raise SystemExit("some examples train on the prompt too; completion_only_loss is not applied")
    print(f"{len(ds)} examples, {steps} optimizer steps, saving at {save_at}, reply tokens {sum(lens):,}", flush=True)

    t0 = time.time()
    trainer.train()
    info = {"student": model_id, "data": args.data, "align": list(sets), "n": args.n, "seed": args.seed,
            "epochs": args.epochs, "lr": args.lr, "effective_batch": args.batch * args.accum, "steps": steps,
            "save_at": save_at, "reply_tokens": sum(lens), "mean_reply_tokens": round(sum(lens) / max(1, len(lens)), 1),
            "minutes": round((time.time() - t0) / 60, 1), "ids": ids,
            "log": [h for h in trainer.state.log_history if "loss" in h]}
    json.dump(info, open(os.path.join(out, "train_info.json"), "w"), indent=1)
    print(f"final loss {info['log'][-1]['loss'] if info['log'] else None}, {info['minutes']} min")
    if os.environ.get("SLURM_JOB_ID"):  # slurm/train.sub evaluates this run's checkpoints after training
        open(os.path.join(OUT, "ckpt", f".run_{os.environ['SLURM_JOB_ID']}"), "w").write(out)
    print("SFT_OK")


if __name__ == "__main__":
    main()
