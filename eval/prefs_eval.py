"""Preference eval: how concentrated is a model's answer distribution on "pick one" questions?

Two modes, both over every paraphrase in eval/categories.yaml:
  sample   N samples per prompt at T=1 (no top-p/top-k), answers normalized (eval/normalize.py).
  logprob  every category: the exact distribution over everything the model would answer. Walks the
           tree of continuations the model itself produces (see run_logprob), so it needs no candidate
           list and no fixed answer format. Each finished branch is normalized like a sample.

Prompt formats (same prompts in both modes, so base and SFT models compare directly):
  chat  the tokenizer's chat template, thinking disabled; a reply ends at the end-of-turn token
  base  "Question: <prompt> <instruction>\nAnswer:"; a reply ends at the first newline

Usage:
  python eval/prefs_eval.py --model Qwen/Qwen3-32B [--name qwen3_32b] [--format auto|chat|base]
         [--modes sample,logprob] [--n 100] [--categories color,letter] [--tp 2] [--max-model-len 2048]
         [--min-prob 1e-4] [--topk 100]
  python eval/prefs_eval.py --model <id> --dry-run     # tokenizer only: print rendered prompts, no GPU
  python eval/prefs_eval.py --model allenai/OLMo-2-0425-1B --format chat --chat-template configs/student_chat.jinja
Writes to $OUT_DIR/eval/prefs/<name>/: samples.jsonl, samples_tree.jsonl.gz (top-k next tokens at every step of
 every sample), logprobs.jsonl, tree.jsonl.gz (every step of the logprob tree: the full top-k next-token
 distribution and where each token leads), summary.json (and prints a table).
"""
import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict

import yaml

from normalize import candidates, first_line_done, normalize

HERE = os.path.dirname(os.path.abspath(__file__))
END_PROBE = "QQZXQ"  # marker used to find the chat template's end-of-turn token


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF id or local checkpoint dir")
    ap.add_argument("--name", help="run name (output dir); default derived from --model")
    ap.add_argument("--format", default="auto", choices=["auto", "chat", "base"])
    ap.add_argument("--chat-template", help="jinja file that replaces the tokenizer's chat template, e.g. "
                    "configs/student_chat.jinja to run a base student in the same format as its SFT checkpoints")
    ap.add_argument("--modes", default="sample,logprob")
    ap.add_argument("--n", type=int, default=100, help="samples per prompt")
    ap.add_argument("--categories", help="comma-separated subset; default all")
    ap.add_argument("--config", default=os.path.join(HERE, "categories.yaml"))
    ap.add_argument("--tp", type=int, help="tensor parallel size; default = visible GPUs")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=32, help="max new tokens per sample / tree depth")
    ap.add_argument("--min-prob", type=float, default=1e-4,
                    help="logprob mode: drop unfinished branches below this probability")
    ap.add_argument("--topk", type=int, default=100, help="logprob mode: next tokens read per branch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", default=os.path.join(os.environ.get("OUT_DIR", "."), "eval", "prefs"))
    ap.add_argument("--dry-run", action="store_true", help="render prompts with the tokenizer only")
    return ap.parse_args()


# ---------- prompts ----------

def load_chat_template(tok, model):
    """Some repos (e.g. Mistral-Small-3.1) keep the template in chat_template.json/.jinja, not the tokenizer."""
    if tok.chat_template:
        return
    from huggingface_hub import hf_hub_download
    for fname in ("chat_template.jinja", "chat_template.json"):
        try:
            path = os.path.join(model, fname) if os.path.isdir(model) else hf_hub_download(model, fname)
        except Exception:
            continue
        if os.path.exists(path):
            text = open(path).read()
            tok.chat_template = json.loads(text)["chat_template"] if fname.endswith(".json") else text
            return


def end_of_turn(tok):
    """Id of the first token the chat template emits after an assistant message (e.g. <|im_end|>)."""
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": END_PROBE}]
    try:
        s = tok.apply_chat_template(msgs, tokenize=False)
    except Exception:  # mistral_common tokenizers refuse a conversation ending on an assistant turn
        return tok.eos_token_id
    suffix_ids = tok(s[s.rindex(END_PROBE) + len(END_PROBE):].lstrip(), add_special_tokens=False).input_ids
    return suffix_ids[0] if suffix_ids else tok.eos_token_id


class Formatter:
    def __init__(self, tok, fmt):
        self.tok, self.fmt = tok, fmt
        self.end_id = end_of_turn(tok) if fmt == "chat" else None
        self.end = tok.decode([self.end_id]) if fmt == "chat" else "\n"
        self.stop_ids = {i for i in (self.end_id, tok.eos_token_id) if i is not None}
        self.newline_ids = None  # filled on first use (decodes the whole vocab, ~seconds)

    def newlines(self):
        if self.newline_ids is None:
            self.newline_ids = {i for i in range(len(self.tok)) if "\n" in self.tok.decode([i])}
        return self.newline_ids

    def prefix_ids(self, text):
        if self.fmt == "chat":
            # Let the tokenizer produce ids: rendering to text and re-encoding turns special tokens into
            # plain text with mistral_common tokenizers (Mistral-Small-3.1 saw a literal "<s>[INST]").
            ids = self.tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                               add_generation_prompt=True, enable_thinking=False)
            if hasattr(ids, "keys"):  # BatchEncoding in transformers 5
                ids = ids["input_ids"]
            return list(ids[0]) if ids and isinstance(ids[0], list) else list(ids)
        return self.tok(f"Question: {text}\nAnswer:").input_ids


def build_prompts(cats, fmt):
    """One entry per (category, paraphrase)."""
    out = []
    for cname, cat in cats.items():
        for i, p in enumerate(cat["prompts"]):
            text = f"{p} {cat['instruction']}"
            out.append({"cat": cname, "pidx": i, "text": text, "ids": fmt.prefix_ids(text)})
    return out


# ---------- metrics ----------

def entropy_bits(probs):
    return max(0.0, -sum(p * math.log2(p) for p in probs if p > 0))


def top(dist, k=10):
    return [[a, round(p, 4)] for a, p in sorted(dist.items(), key=lambda x: -x[1])[:k]]


def sample_metrics(rows):
    """rows: dicts with answer/valid. Distribution is over valid answers only."""
    valid = Counter(r["answer"] for r in rows if r["valid"])
    nv = sum(valid.values())
    dist = {a: c / nv for a, c in valid.items()} if nv else {}
    invalid = Counter(r["answer"] or "<empty>" for r in rows if not r["valid"])
    return {"n": len(rows), "valid_rate": round(nv / len(rows), 4) if rows else 0.0,
            "n_distinct": len(valid), "entropy_bits": round(entropy_bits(dist.values()), 4),
            "top1_share": round(max(dist.values()), 4) if dist else None, "top": top(dist),
            "top_invalid": invalid.most_common(5)}


def prob_metrics(weights, cat):
    """weights: {(answer, valid): probability}. Same fields as sample_metrics, with probability mass in
    place of counts: the distribution is renormalized over valid answers."""
    valid = {a: w for (a, ok), w in weights.items() if ok}
    total, nv = sum(weights.values()), sum(valid.values())
    dist = {a: w / nv for a, w in valid.items()} if nv else {}
    invalid = sorted(((a or "<empty>", w) for (a, ok), w in weights.items() if not ok), key=lambda x: -x[1])
    m = {"valid_rate": round(nv / total, 4) if total else 0.0, "n_distinct": len(dist),
         "entropy_bits": round(entropy_bits(dist.values()), 4),
         "top1_share": round(max(dist.values()), 4) if dist else None, "top": top(dist),
         "top_invalid": [[a, round(w / total, 4)] for a, w in invalid[:5]]}
    if candidates(cat):
        m["max_entropy_bits"] = round(math.log2(len(candidates(cat))), 4)
    return m


# ---------- modes ----------

def run_sample(llm, prompts, cats, args, fmt):
    """N samples per prompt. Also logs, at every step of every sample, the model's top-k next tokens.
    Samples that share a prefix share a step, so the log is a prefix tree (see write_jsonl_gz callers):
    one record per distinct (prompt, prefix), with how many samples passed through it and took each token.
    Returns (sample rows, summary, sample tree, drift); drift is the largest difference between the
    log-probs two samples saw at the same step (should be ~0: same model, same prefix)."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    tok = fmt.tok
    params = SamplingParams(n=args.n, temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, seed=args.seed,
                            stop=["\n"] if fmt.fmt == "base" else None, logprobs=args.topk)
    outs = llm.generate([TokensPrompt(prompt_token_ids=p["ids"]) for p in prompts], params, use_tqdm=True)
    rows, steps, drift = [], {}, 0.0  # steps: (prompt index, prefix ids) -> record
    for pi, (p, o) in enumerate(zip(prompts, outs)):
        for c in o.outputs:
            ans, ok = normalize(c.text, cats[p["cat"]])
            ids, path = list(c.token_ids), []
            for j, t in enumerate(ids):
                lps = c.logprobs[j]
                key = (pi, tuple(ids[:j]))
                n = steps.get(key)
                if n is None:
                    n = steps[key] = {"id": len(steps), "cat": p["cat"], "pidx": p["pidx"], "depth": j, "ids": ids[:j],
                                      "count": 0, "next": [[tid, tok.decode([tid]), round(l.logprob, 6), 0]
                                                           for tid, l in sorted(lps.items(), key=lambda x: -x[1].logprob)]}
                n["count"] += 1
                for e in n["next"]:
                    if e[0] == t:
                        e[3] += 1
                        drift = max(drift, abs(e[2] - lps[t].logprob))
                        break
                else:  # cannot happen: vLLM always returns the sampled token's log-prob
                    n["next"].append([t, tok.decode([t]), round(lps[t].logprob, 6), 1])
                path.append(n["id"])
            rows.append({"cat": p["cat"], "pidx": p["pidx"], "raw": c.text, "answer": ans, "valid": ok,
                         "finish": c.finish_reason, "token_ids": ids, "steps": path})
    by_cat, by_prompt = defaultdict(list), defaultdict(list)
    for r in rows:
        by_cat[r["cat"]].append(r)
        by_prompt[(r["cat"], r["pidx"])].append(r)
    summary = {}
    for cname, rs in by_cat.items():
        m = sample_metrics(rs)
        per = [sample_metrics(by_prompt[(cname, i)]) for i in range(len(cats[cname]["prompts"]))]
        m["mean_prompt_entropy_bits"] = round(sum(x["entropy_bits"] for x in per) / len(per), 4)
        m["prompt_top1"] = [x["top"][0] if x["top"] else None for x in per]
        m["truncated_rate"] = round(sum(r["finish"] == "length" for r in rs) / len(rs), 4)
        summary[cname] = m
    return rows, summary, list(steps.values()), drift


def run_logprob(llm, prompts, cats, args, fmt):
    """Exact answer distribution, by walking the tree of the model's own continuations.

    Each prompt starts as one branch with probability 1. Every step sends all open branches to vLLM in
    one batch and splits each branch's probability over its top-k next tokens. A branch finishes when
      - the model ends its reply (end-of-turn / EOS; newline in base format),
      - its first line is complete (normalize() reads only the first line, so the answer is fixed), or
      - it reaches --max-tokens, the same cap as sample mode.
    Open branches below --min-prob are dropped, as is the mass outside the top-k tokens. Both are
    reported as `pruned`. Finished branches are normalized like samples, so this is exactly the
    distribution that sample mode draws from (T=1, no top-p/top-k), without sampling noise.

    Returns (leaf rows, summary, tree). `tree` has one record per expanded branch (see write_tree): the
    model's full top-k next-token distribution at that point and the answers each token leads to.
    """
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    tok, newlines = fmt.tok, fmt.newlines()
    # T=1 and no truncation, so the returned logprobs are the model's own distribution.
    params = SamplingParams(max_tokens=1, temperature=1.0, top_p=1.0, logprobs=args.topk, seed=args.seed)
    nodes = {}  # node id -> expanded-branch record
    next_id = len(prompts)  # roots are ids 0..len(prompts)-1
    frontier = [(i, [], 0.0, i, None) for i in range(len(prompts))]  # (prompt, ids, log P, node id, parent)
    leaves, pruned = [], [0.0] * len(prompts)
    for depth in range(1, args.max_tokens + 1):
        if not frontier:
            break
        t0 = time.time()
        outs = llm.generate([TokensPrompt(prompt_token_ids=prompts[i]["ids"] + ids) for i, ids, *_ in frontier],
                            params, use_tqdm=False)
        nxt = []
        for (i, ids, lp, nid, parent), o in zip(frontier, outs):
            kept, entries = 0.0, []
            for t, l in sorted(o.outputs[0].logprobs[0].items(), key=lambda x: -x[1].logprob):
                clp = lp + l.logprob
                p = math.exp(clp)
                child, end = next_id, None
                if t in fmt.stop_ids:
                    leaves.append((i, ids, clp, "eos", child, nid))
                    end = "eos"
                elif t in newlines and (fmt.fmt == "base" or first_line_done(tok.decode(ids + [t]))):
                    leaves.append((i, ids + [t], clp, "newline", child, nid))
                    end = "newline"
                elif p < args.min_prob:
                    child, end = None, "pruned"
                elif depth == args.max_tokens:
                    leaves.append((i, ids + [t], clp, "length", child, nid))
                    end = "length"
                else:
                    nxt.append((i, ids + [t], clp, child, nid))
                    end = "open"
                if child is not None:
                    next_id += 1
                    kept += p
                # [token id, token text, log P(token | this branch), how the child ended, child node id]
                entries.append([t, tok.decode([t]), round(l.logprob, 6), end, child])
            pruned[i] += max(0.0, math.exp(lp) - kept)
            nodes[nid] = {"id": nid, "parent": parent, "cat": prompts[i]["cat"], "pidx": prompts[i]["pidx"],
                          "depth": depth - 1, "logp": round(lp, 6), "prefix": tok.decode(ids), "next": entries}
        print(f"logprob depth {depth}: {len(frontier)} branches expanded, {len(nxt)} open, "
              f"{len(leaves)} finished ({time.time() - t0:.0f}s)", flush=True)
        frontier = nxt

    rows = []
    weights = defaultdict(lambda: defaultdict(float))  # (cat, pidx) -> {(answer, valid): probability}
    leaf_answer = {}  # leaf node id -> (answer, valid)
    for i, ids, lp, end, nid, parent in leaves:
        p = prompts[i]
        text = tok.decode(ids, skip_special_tokens=True)
        if fmt.fmt == "base":
            text = text.split("\n", 1)[0]
        ans, ok = normalize(text, cats[p["cat"]])
        leaf_answer[nid] = (ans, ok)
        rows.append({"cat": p["cat"], "pidx": p["pidx"], "raw": text, "answer": ans, "valid": ok,
                     "prob": math.exp(lp), "end": end, "node": nid, "parent": parent})
        weights[(p["cat"], p["pidx"])][(ans, ok)] += math.exp(lp)
    rows.sort(key=lambda r: (r["cat"], r["pidx"], -r["prob"]))
    tree = annotate_tree(nodes, leaf_answer, prompts)

    summary = {}
    for cname in dict.fromkeys(p["cat"] for p in prompts):  # config order
        idx = [i for i, p in enumerate(prompts) if p["cat"] == cname]
        per = [weights[(cname, prompts[i]["pidx"])] for i in idx]
        pooled = defaultdict(float)  # mean over paraphrases, like pooling equal-sized sample sets
        for w in per:
            for k, v in w.items():
                pooled[k] += v / len(per)
        m = prob_metrics(pooled, cats[cname])
        pm = [prob_metrics(w, cats[cname]) for w in per]
        m["mean_prompt_entropy_bits"] = round(sum(x["entropy_bits"] for x in pm) / len(pm), 4)
        m["prompt_top1"] = [x["top"][0] if x["top"] else None for x in pm]
        m["truncated_rate"] = round(sum(r["prob"] for r in rows if r["cat"] == cname and r["end"] == "length")
                                    / len(idx), 4)
        m["pruned_mean"] = round(sum(pruned[i] for i in idx) / len(idx), 4)
        m["pruned_max"] = round(max(pruned[i] for i in idx), 4)
        m["n_leaves"] = sum(r["cat"] == cname for r in rows)
        summary[cname] = m
    return rows, summary, tree


def annotate_tree(nodes, leaf_answer, prompts):
    """Adds to every expanded node the answers reachable below it, and to each of its next-token entries
    the answer that token leads to. Probabilities are conditional on reaching the node; mass on pruned
    branches is reported as `dropped`, so `answers` plus `dropped` sums to 1."""
    below = {}  # node id -> {answer: P(answer | node)}, answer None = dropped branch

    def dist(nid):  # children always have larger ids, so a reverse sweep sees them first
        if nid in leaf_answer:
            ans, ok = leaf_answer[nid]
            return {(ans if ok else f"[invalid] {ans or '<empty>'}"): 1.0}
        return below[nid]

    for nid in sorted(nodes, reverse=True):
        acc = defaultdict(float)
        for e in nodes[nid]["next"]:
            pc = math.exp(e[2])
            if e[4] is None:
                acc[None] += pc
                e.append(None)
                continue
            d = dist(e[4])
            for a, v in d.items():
                acc[a] += pc * v
            best = max(((a, v) for a, v in d.items() if a is not None), key=lambda x: x[1], default=(None, 0.0))
            e.append([best[0], round(best[1], 4)])  # [answer this token mostly leads to, its share]
        acc[None] += max(0.0, 1.0 - sum(acc.values()))  # mass outside the top-k
        below[nid] = dict(acc)
        ranked = sorted(((a, v) for a, v in acc.items() if a is not None), key=lambda x: -x[1])
        nodes[nid]["answers"] = [[a, round(v, 6)] for a, v in ranked if v >= 1e-6]
        nodes[nid]["dropped"] = round(acc[None], 6)
    for i, p in enumerate(prompts):
        if i in nodes:
            nodes[i]["prompt"] = p["text"]
    return [nodes[k] for k in sorted(nodes)]


def write_jsonl_gz(path, records):
    import gzip
    with gzip.open(path, "wt") as f:
        for n in records:
            f.write(json.dumps(n, ensure_ascii=False) + "\n")


def write_tree(path, tree):
    """tree.jsonl.gz, one line per expanded branch:
      id, parent      node ids (root = prompt index; parent None)
      cat, pidx       question and wording
      depth, prefix   tokens generated so far and their text
      logp            log P(prefix | prompt)
      next            the full top-k next-token distribution here, likeliest first:
                      [token id, token text, log P(token | prefix), end, child id, [answer, share]]
                      end: open (expanded further), eos, newline, length, or pruned (below --min-prob)
                      [answer, share]: the answer this token mostly leads to (None if pruned)
      answers         [[answer, P(answer | prefix)], ...] over everything reachable from here
      dropped         P(pruned or outside top-k | prefix)
      prompt          (roots only) the prompt text
    """
    write_jsonl_gz(path, tree)


# ---------- main ----------

def print_table(summary):
    print(f"\n{'category':18} {'mode':8} {'H bits':>7} {'H max':>6} {'top1':>6}  {'valid':>6} {'pruned':>6}  top answers")
    for cname in summary.get("sample", {}) | summary.get("logprob", {}):
        for mode in ("sample", "logprob"):
            m = summary.get(mode, {}).get(cname)
            if not m:
                continue
            hmax = m.get("max_entropy_bits", "")
            pruned = f"{m['pruned_mean']:6.3f}" if mode == "logprob" else ""
            tops = ", ".join(f"{a} {p:.0%}" for a, p in m["top"][:4])
            print(f"{cname:18} {mode:8} {m['entropy_bits']:7.2f} {hmax:>6} {m['top1_share'] or 0:6.2f}  "
                  f"{m['valid_rate']:6.2f} {pruned:>6}  {tops}")


def main():
    args = parse_args()
    from transformers import AutoTokenizer
    cats = yaml.safe_load(open(args.config))
    if args.categories:
        cats = {k: cats[k] for k in args.categories.split(",")}
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.chat_template:
        tok.chat_template = open(args.chat_template).read()
    load_chat_template(tok, args.model)
    fmt_name = args.format if args.format != "auto" else ("chat" if tok.chat_template else "base")
    fmt = Formatter(tok, fmt_name)
    prompts = build_prompts(cats, fmt)
    modes = args.modes.split(",")
    name = args.name or args.model.rstrip("/").replace("/", "__")
    print(f"model={args.model} format={fmt_name} end_marker={fmt.end!r} prompts={len(prompts)} modes={modes}")

    if args.dry_run:
        seen = set()
        for p in prompts:
            if p["cat"] in seen:
                continue
            seen.add(p["cat"])
            special = set(tok.all_special_ids) | {fmt.end_id}
            n_special = sum(i in special for i in p["ids"])
            print(f"\n--- {p['cat']} ({len(p['ids'])} prefix tokens, {n_special} special)\n{tok.decode(p['ids'])!r}")
        print(f"\nstop tokens: {[tok.decode([i]) for i in fmt.stop_ids]}, newline tokens: {len(fmt.newlines())}")
        return

    from vllm import LLM
    import torch
    tp = args.tp or torch.cuda.device_count()
    t0 = time.time()
    llm = LLM(model=args.model, tensor_parallel_size=tp, max_model_len=args.max_model_len,
              gpu_memory_utilization=0.90, language_model_only=True, generation_config="vllm", seed=args.seed,
              max_logprobs=max(args.topk, 20))
    print(f"loaded in {time.time() - t0:.0f}s (TP={tp})", flush=True)

    out = os.path.join(args.out_root, name)
    os.makedirs(out, exist_ok=True)
    # A run with only some modes keeps the other modes' results from an earlier run in the same folder.
    spath = os.path.join(out, "summary.json")
    summary = json.load(open(spath)) if os.path.exists(spath) else {}
    summary.update({"model": args.model, "format": fmt_name, "end_marker": fmt.end, "n": args.n, "seed": args.seed,
                    "min_prob": args.min_prob, "topk": args.topk, "stop_ids": sorted(fmt.stop_ids)})
    if "sample" in modes:
        rows, summary["sample"], steps, drift = run_sample(llm, prompts, cats, args, fmt)
        summary["sample_steps"] = {"n_steps": len(steps), "max_logprob_drift": round(drift, 6)}
        print(f"sample steps logged: {len(steps)}, max log-prob drift between samples: {drift:.2e}", flush=True)
        with open(os.path.join(out, "samples.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        # samples_tree.jsonl.gz, one line per distinct step of the samples: id, cat, pidx, depth, ids (prefix
        # token ids), count (samples through it), next: the top-k next tokens [token id, text, log P, n taken].
        # samples.jsonl rows have token_ids and steps (the step id at each token).
        write_jsonl_gz(os.path.join(out, "samples_tree.jsonl.gz"), steps)
    if "logprob" in modes:
        rows, summary["logprob"], tree = run_logprob(llm, prompts, cats, args, fmt)
        with open(os.path.join(out, "logprobs.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        write_tree(os.path.join(out, "tree.jsonl.gz"), tree)
    with open(spath, "w") as f:
        json.dump(summary, f, indent=1)
    print_table(summary)
    print(f"\nwrote {out}\nPREFS_EVAL_OK")


if __name__ == "__main__":  # vLLM spawns workers that re-import this file
    main()
