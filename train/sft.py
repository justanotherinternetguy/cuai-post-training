"""Instruction fine-tuning (SFT) of a base student on one data set (a teacher's replies, or the human replies).

Runs that are compared must see the same prompts: --align lists every data set in the comparison, and only
prompts that all of them answered with a finished reply (finish == "stop") are used, in the order of
$OUT_DIR/data/prompts/train.jsonl, first --n of them. Hyperparameters are the same for every data set.

Format: configs/student_chat.jinja, loss on reply tokens only (prompt/completion data). The chat template
is saved with every checkpoint, so prefs_eval.py --format chat runs the student in its training format.

Multi-GPU: run under torchrun (slurm/train.sub does this when it has more than one GPU). The model is then
sharded with FSDP2 (full shard, one unit per decoder layer, activation checkpointing), loaded on rank 0 only
and broadcast. The effective batch stays --effective-batch (32) whatever the GPU count, so the optimizer
steps and checkpoint steps are the same as on one GPU. A 7B student in fp32 (weights, grads, Adam: ~112 GB)
needs 4x 48 GB GPUs this way; --batch 2 keeps activations and logits under 48 GB at 4096 tokens.

Usage:
  python train/sft.py --student olmo2_1b --data gemma3_27b --align gemma3_27b,olmo2_32b,human --n 5000
  torchrun --nproc_per_node 4 train/sft.py --student olmo3_7b --batch 2 --save-dtype bf16 --data ... --align ...
Writes $OUT_DIR/ckpt/<student>/<data>_n<n>_s<seed>/checkpoint-<step> at --save-fracs of training,
plus train_info.json (prompts used, reply tokens, loss curve).
"""
import argparse
import datetime
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
    ap.add_argument("--effective-batch", type=int, default=32,
                    help="batch * GPUs * gradient accumulation; the accumulation is derived from it")
    ap.add_argument("--max-length", type=int, default=4096, help="prompt + reply tokens (teacher replies go up to 3072)")
    ap.add_argument("--save-fracs", default="0.05,0.1,0.25,0.5,1.0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chat-template", default=os.path.join(ROOT, "configs", "student_chat.jinja"))
    ap.add_argument("--save-dtype", default="fp32", choices=["fp32", "bf16"],
                    help="checkpoint weights; bf16 halves the disk use (a 7B checkpoint: 15 GB instead of 29 GB)")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()
    world = int(os.environ.get("WORLD_SIZE", 1))
    if args.effective_batch % (args.batch * world):
        raise SystemExit(f"--effective-batch {args.effective_batch} is not a multiple of batch {args.batch} x {world} GPUs")
    accum = args.effective_batch // (args.batch * world)

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
    from accelerate import PartialState
    from datasets import Dataset
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    fsdp = world > 1
    if fsdp:
        os.environ["ACCELERATE_USE_FSDP"] = "true"
        os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "true"
    # Starts torch.distributed under torchrun. Long timeout: ranks > 0 wait in a broadcast while rank 0 loads
    # the weights (several minutes for a 7B model from the shared disk).
    state = PartialState(**({"timeout": datetime.timedelta(minutes=60)} if fsdp else {}))
    main = state.is_main_process

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.chat_template = open(args.chat_template).read()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = Dataset.from_list([{"prompt": [{"role": "user", "content": r["prompt"]}],
                             "completion": [{"role": "assistant", "content": r["reply"]}]} for r in rows])

    run = f"{args.data}_n{args.n}_s{args.seed}"
    out = os.path.join(OUT, "ckpt", args.student, run)
    os.makedirs(out, exist_ok=True)
    steps = math.ceil(math.ceil(args.n / (args.batch * world)) / accum * args.epochs)
    save_at = sorted({max(1, round(f * steps)) for f in map(float, args.save_fracs.split(","))})

    class SaveAt(TrainerCallback):
        def on_step_end(self, a, state, control, **kw):
            if state.global_step in save_at:
                control.should_save = True

    save_dtype = torch.bfloat16 if args.save_dtype == "bf16" else None

    class Trainer(SFTTrainer):
        def _save(self, output_dir=None, state_dict=None):
            if save_dtype is not None:  # only the main process gets here, with the full (gathered) weights
                if state_dict is None:
                    state_dict = self.model.state_dict()
                state_dict = {k: v.to(save_dtype) if v.is_floating_point() else v for k, v in state_dict.items()}
            super()._save(output_dir, state_dict=state_dict)
            if save_dtype is not None:
                path = os.path.join(output_dir, "config.json")
                c = json.load(open(path))
                c.update({k: "bfloat16" for k in ("dtype", "torch_dtype") if k in c} or {"dtype": "bfloat16"})
                json.dump(c, open(path, "w"), indent=2)

    # FSDP2: shard every decoder layer (the model's _no_split_modules), checkpoint activations per layer
    # (the FSDP form of gradient checkpointing), gather full weights on rank 0 to save.
    fsdp_args = dict(fsdp="full_shard auto_wrap",
                     fsdp_config={"version": 2, "activation_checkpointing": True, "cpu_ram_efficient_loading": True,
                                  "state_dict_type": "FULL_STATE_DICT"}) if fsdp else {}
    cfg = SFTConfig(output_dir=out, run_name=f"{args.student}/{run}", num_train_epochs=args.epochs,
                    learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=0.03, weight_decay=0.0,
                    per_device_train_batch_size=args.batch, gradient_accumulation_steps=accum,
                    max_length=args.max_length, completion_only_loss=True, packing=False, bf16=True,
                    gradient_checkpointing=not fsdp, logging_steps=10, save_strategy="no", save_only_model=True,
                    seed=args.seed, data_seed=args.seed, report_to=["wandb"] if args.wandb and main else [],
                    dataloader_num_workers=2, **fsdp_args)
    if fsdp and not main:
        # transformers 5.20 reads the full checkpoint on every rank, even with cpu_ram_efficient_loading (ranks > 0
        # then swap in zeros of the same size): 4 x 29 GB of host RAM for a 7B student, OOM at --mem=128g
        # (jobs 152537-152545). So only rank 0 loads. The other ranks build the model with uninitialized
        # weights; to_empty never touches the pages, so they cost no RAM. Accelerate's FSDP2 prepare
        # (cpu_ram_efficient_loading) then broadcasts rank 0's weights into every shard.
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_id), dtype=torch.float32)
        model.to_empty(device="cpu")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)  # fp32 master weights, bf16 autocast
    if fsdp:  # accelerate does not broadcast buffers (e.g. rotary inv_freq): copy them from rank 0
        for _, b in model.named_buffers():
            t = b.to(state.device)
            torch.distributed.broadcast(t, src=0)
            b.copy_(t.cpu())
    trainer = Trainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, callbacks=[SaveAt()])

    # Check the loss mask: only the reply (and its end token) is trained on. TRL >= 1.x folds the completion
    # mask into `labels` (-100 = not trained) and drops the completion_mask column.
    ex = trainer.train_dataset[0]
    if "labels" not in ex:
        raise SystemExit(f"no labels column (columns: {trainer.train_dataset.column_names}); loss mask unknown")
    lens = [sum(l != -100 for l in labels) for labels in trainer.train_dataset["labels"]]
    if any(n == len(ids) for n, ids in zip(lens, trainer.train_dataset["input_ids"])):
        raise SystemExit("some examples train on the prompt too; completion_only_loss is not applied")
    if main:
        print("example input:", repr(tok.decode(ex["input_ids"])[-300:]))
        print("trained part: ", repr(tok.decode([t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100])[-200:]))
        print("untrained end:", repr(tok.decode([t for t, l in zip(ex["input_ids"], ex["labels"]) if l == -100])[-200:]))
        print(f"{len(ds)} examples, {world} GPU(s) x batch {args.batch} x accum {accum} = {args.effective_batch}, "
              f"{steps} optimizer steps, saving at {save_at}, reply tokens {sum(lens):,}", flush=True)

    t0 = time.time()
    trainer.train()
    if not main:
        return
    info = {"student": model_id, "data": args.data, "align": list(sets), "n": args.n, "seed": args.seed,
            "epochs": args.epochs, "lr": args.lr, "effective_batch": args.effective_batch, "gpus": world,
            "per_device_batch": args.batch, "save_dtype": args.save_dtype, "steps": steps,
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
