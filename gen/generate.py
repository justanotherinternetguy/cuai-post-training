"""Answer the IFT prompts with one model: one reply per prompt, sampled at T=1 with no top-p / top-k, so the
replies are a fair sample of the model's own distribution (a lower temperature would collapse them further).

Used for teachers (the SFT data) and for students (next-token entropy on held-out prompts).

Usage:
  python gen/generate.py --model google/gemma-3-27b-it --name gemma3_27b --splits train:5500,heldout:500
  python gen/generate.py --model <student ckpt> --name <run> --splits heldout:500 --out-root $OUT_DIR/eval/gen
         [--chat-template configs/student_chat.jinja]
Reads $OUT_DIR/data/prompts/<split>.jsonl (first N rows of each split).
Writes <out-root>/<name>/<split>.jsonl: {id, prompt, reply, finish, n_tokens, entropy}
  finish: stop (end of turn), length (hit --max-tokens), skipped (prompt too long)
  entropy: mean next-token entropy over the reply in bits, from the top-20 log-probs at each step; the mass
           outside the top 20 counts as one outcome, so it is a lower bound
and <out-root>/<name>/<split>.stats.json.

Prompts go to vLLM in chunks of --chunk. Each finished chunk is appended to <split>.jsonl.partial, so RAM stays
flat (vLLM keeps every reply's log-probs until a generate call returns) and a killed or requeued job resumes
from the partial file. <split>.jsonl is written only when the split is complete.
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "eval"))
from prefs_eval import Formatter, load_chat_template  # noqa: E402

DATA = os.path.join(os.environ.get("OUT_DIR", "."), "data")
TOPK = 20


def step_entropy(lp):
    """Entropy in bits of one step from its top-k log-probs (dict token id -> Logprob)."""
    ps = [math.exp(x.logprob) for x in lp.values()]
    rest = max(0.0, 1.0 - sum(ps))
    return -sum(p * math.log2(p) for p in ps + [rest] if p > 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--splits", default="train:5500,heldout:500", help="split:N,... (first N prompts of each)")
    ap.add_argument("--chat-template", help="jinja file replacing the tokenizer's chat template")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--tp", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=1000, help="prompts per generate call (written to disk after each)")
    ap.add_argument("--out-root", default=DATA)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.chat_template:
        tok.chat_template = open(args.chat_template).read()
    load_chat_template(tok, args.model)
    fmt = Formatter(tok, "chat")

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    import torch
    t0 = time.time()
    # prefetch: read the checkpoint files into page cache in the background. vLLM does this only on NFS/Lustre,
    # and the cluster's model dir is AUTOFS, where loading without it took ~1 h for a 61 GB model.
    llm = LLM(model=args.model, tensor_parallel_size=args.tp or torch.cuda.device_count(),
              max_model_len=args.max_model_len, gpu_memory_utilization=0.90, language_model_only=True,
              generation_config="vllm", seed=args.seed, max_logprobs=TOPK, safetensors_load_strategy="prefetch")
    print(f"loaded in {time.time() - t0:.0f}s, end of turn {fmt.end!r}, stop ids {sorted(fmt.stop_ids)}", flush=True)
    params = SamplingParams(n=1, temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, seed=args.seed,
                            stop_token_ids=sorted(fmt.stop_ids), logprobs=TOPK)

    out_dir = os.path.join(args.out_root, args.name)
    os.makedirs(out_dir, exist_ok=True)
    for spec in args.splits.split(","):
        split, n = spec.split(":")
        rows = [json.loads(line) for line in open(os.path.join(DATA, "prompts", f"{split}.jsonl"))][:int(n)]
        ids = [fmt.prefix_ids(r["prompt"]) for r in rows]
        ok = [i for i, x in enumerate(ids) if len(x) + args.max_tokens <= args.max_model_len]
        final = os.path.join(out_dir, f"{split}.jsonl")
        part = final + ".partial"
        have = {}  # id -> record, for prompts already answered
        if os.path.exists(part):
            for line in open(part):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:  # last line cut off when the job was killed
                    break
                have[rec["id"]] = rec
            with open(part, "w") as f:  # drop a cut-off last line before appending
                f.writelines(json.dumps(rec) + "\n" for rec in have.values())
        resumed = len(have)
        todo = [i for i in ok if rows[i]["id"] not in have]
        print(f"{split}: {len(ok)} prompts fit, {resumed} already done, {len(todo)} to go", flush=True)
        t1 = time.time()
        n_new = 0
        with open(part, "a") as f:
            for c in range(0, len(todo), args.chunk):
                idx = todo[c:c + args.chunk]
                outs = llm.generate([TokensPrompt(prompt_token_ids=ids[i]) for i in idx], params, use_tqdm=True)
                for i, out in zip(idx, outs):
                    o = out.outputs[0]
                    hs = [step_entropy(lp) for lp in (o.logprobs or []) if lp]
                    rec = {"id": rows[i]["id"], "prompt": rows[i]["prompt"], "reply": o.text.strip(),
                           "finish": "length" if o.finish_reason == "length" else "stop",
                           "n_tokens": len(o.token_ids), "entropy": round(sum(hs) / len(hs), 4) if hs else None}
                    have[rec["id"]] = rec
                    n_new += rec["n_tokens"]
                    f.write(json.dumps(rec) + "\n")
                del outs
                f.flush()
                os.fsync(f.fileno())
                print(f"{split}: {len(have)}/{len(ok)} done, {time.time() - t1:.0f}s", flush=True)
        dt = time.time() - t1
        recs = [have.get(r["id"]) or {"id": r["id"], "prompt": r["prompt"], "reply": None, "finish": "skipped",
                                      "n_tokens": 0, "entropy": None} for r in rows]
        with open(final + ".tmp", "w") as f:
            f.writelines(json.dumps(rec) + "\n" for rec in recs)
        os.replace(final + ".tmp", final)
        os.remove(part)
        done = [r for r in recs if r["finish"] == "stop"]
        stats = {"model": args.model, "split": split, "n": len(recs),
                 "finish": {k: sum(r["finish"] == k for r in recs) for k in ("stop", "length", "skipped")},
                 "mean_tokens_stop": round(sum(r["n_tokens"] for r in done) / max(1, len(done)), 1),
                 "mean_entropy_stop": round(sum(r["entropy"] for r in done if r["entropy"] is not None)
                                            / max(1, len(done)), 4),
                 "tokens": sum(r["n_tokens"] for r in recs), "resumed": resumed,
                 "seconds": round(dt), "tok_per_s": round(n_new / max(dt, 1e-9))}
        json.dump(stats, open(os.path.join(out_dir, f"{split}.stats.json"), "w"), indent=1)
        print(json.dumps(stats), flush=True)
    print("GEN_OK")


if __name__ == "__main__":
    main()
