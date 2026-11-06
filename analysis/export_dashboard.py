"""Turn preference-eval outputs (logprob tree mode) into the JSON files the answer-distribution dashboard reads.

Usage:
  python analysis/export_dashboard.py <prefs dir> <out dir> [model ...]
<prefs dir> holds one folder per model (summary.json, logprobs.jsonl, tree.jsonl.gz), e.g. a copy of
/share/cuvl/qx235/unicorn/eval/prefs. Writes to <out dir>:
  models.json categories.json answers.json prompts.json prompt_answers.json replies.json samples.json
      one table each, all models together (column `model`); samples.json: every distinct sampled reply per
      wording with how many of the 100 samples gave it, and `last`, the id of its final token in the tree
  tree_<model>.json
      one row per step of the token tree that the page can show (see tree_rows)
"""
import gzip
import json
import math
import os
import sys
from collections import defaultdict

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "eval"))
from normalize import candidates, first_line_done  # noqa: E402

LABELS = {"number_1_10": "Number 1–10", "number_1_100": "Number 1–100", "vacation_country": "Vacation country",
          "city": "City", "color": "Favorite color", "animal": "Animal", "first_name": "First name",
          "letter": "Letter", "fruit": "Fruit", "coin_flip": "Coin flip"}
MODEL_LABELS = {"qwen25_32b": "Qwen2.5-32B", "qwen3_32b": "Qwen3-32B", "olmo2_32b": "OLMo-2-32B",
                "gemma3_27b": "Gemma-3-27B", "mistral31_24b": "Mistral-3.1-24B"}
MIN_SHARE = 0.0005    # prompt_answers: shares below this are left out
N_REPLIES = 25        # replies: likeliest exact replies kept per wording
# tree: (min_p, full_p) levels, tried in order until a model's tree file fits TREE_MAX_BYTES. Steps reached
# with P < min_p are left out (the first level is the eval's --min-prob); answer-choosing steps reached with
# P >= full_p keep all top-k tokens. Models with spread-out answers (OLMo) need a higher level.
TREE_LEVELS = [(1e-4, 1e-3), (3e-4, 3e-3), (1e-3, 1e-2), (3e-3, 3e-2), (1e-2, 1e-1)]
TREE_MAX_BYTES = 6_000_000
TREE_MIN_COND = 1e-2  # tree: other steps keep tokens the reply goes on with, plus others with P(token | step) >= this


def entropy(ps):
    return max(0.0, -sum(p * math.log2(p) for p in ps if p > 0))


def r4(x):
    return None if x is None else round(x, 4)


def load_model(d):
    summary = json.load(open(os.path.join(d, "summary.json")))
    leaves = [json.loads(line) for line in open(os.path.join(d, "logprobs.jsonl"))]
    with gzip.open(os.path.join(d, "tree.jsonl.gz"), "rt") as f:
        tree = [json.loads(line) for line in f]
    return summary, leaves, tree


def tables(model, summary, leaves, cats):
    lp = summary["logprob"]
    per = defaultdict(lambda: defaultdict(float))       # (cat, pidx) -> {answer: prob}, valid only
    total = defaultdict(float)                           # (cat, pidx) -> total kept prob
    for r in leaves:
        total[(r["cat"], r["pidx"])] += r["prob"]
        if r["valid"]:
            per[(r["cat"], r["pidx"])][r["answer"]] += r["prob"]
    cat_rows, ans_rows, prompt_rows, pa_rows, reply_rows = [], [], [], [], []
    for c, m in lp.items():
        cat = cats[c]
        n = len(cat["prompts"])
        pooled = defaultdict(float)
        for i in range(n):
            w = per[(c, i)]
            s = sum(w.values())
            dist = {a: v / s for a, v in w.items()} if s else {}
            for a, v in dist.items():
                pooled[a] += v / n
            top = max(dist.items(), key=lambda x: x[1], default=(None, None))
            prompt_rows.append({"model": model, "category": c, "pidx": i, "prompt": f"{cat['prompts'][i]} {cat['instruction']}",
                                "top_answer": top[0], "top_share": r4(top[1]), "entropy_bits": r4(entropy(dist.values())),
                                "valid_rate": r4(s / total[(c, i)]) if total[(c, i)] else None})
            pa_rows += [{"model": model, "category": c, "pidx": i, "answer": a, "share": r4(v)}
                        for a, v in sorted(dist.items(), key=lambda x: -x[1]) if v >= MIN_SHARE]
        ranked = sorted(pooled.items(), key=lambda x: -x[1])
        ans_rows += [{"model": model, "category": c, "answer": a, "share": round(v, 6), "rank": k + 1}
                     for k, (a, v) in enumerate(ranked)]
        cands = candidates(cat)
        cat_rows.append({"model": model, "category": c, "label": LABELS.get(c, c), "question": cat["prompts"][0],
                         "instruction": cat["instruction"], "kind": cat["kind"],
                         "entropy_bits": m["entropy_bits"], "max_entropy_bits": r4(math.log2(len(cands))) if cands else None,
                         "mean_prompt_entropy_bits": m["mean_prompt_entropy_bits"],
                         "top_answer": ranked[0][0] if ranked else None, "top_share": r4(ranked[0][1]) if ranked else None,
                         "valid_rate": m["valid_rate"], "pruned_mean": m["pruned_mean"], "n_distinct": len(ranked),
                         "sample_entropy_bits": summary.get("sample", {}).get(c, {}).get("entropy_bits")})
    by_prompt = defaultdict(list)
    for r in leaves:
        by_prompt[(r["cat"], r["pidx"])].append(r)
    for (c, i), rs in by_prompt.items():
        for r in sorted(rs, key=lambda r: -r["prob"])[:N_REPLIES]:
            reply_rows.append({"model": model, "id": f"{model}.{r['node']}", "category": c, "pidx": i, "reply": r["raw"],
                               "answer": r["answer"] or "<empty>", "counts": r["valid"], "chance": round(r["prob"], 6),
                               "ended_by": {"eos": "end of turn", "newline": "line break", "length": "32-token cap"}[r["end"]]})
    return cat_rows, ans_rows, prompt_rows, pa_rows, reply_rows


KIND = {"open": "token", "eos": "end", "newline": "stop", "length": "stop", "pruned": "cut"}


def load_samples(d):
    """Distinct sampled replies per wording (same token ids = same path), with the logged sample steps."""
    path = os.path.join(d, "samples_tree.jsonl.gz")
    rows = [json.loads(line) for line in open(os.path.join(d, "samples.jsonl"))]
    if not os.path.exists(path) or "token_ids" not in rows[0]:
        return [], {}
    with gzip.open(path, "rt") as f:
        steps = {n["id"]: n for n in (json.loads(line) for line in f)}
    groups = {}
    for r in rows:
        k = (r["cat"], r["pidx"], tuple(r["token_ids"]))
        if k not in groups:
            groups[k] = dict(r, count=0)
        groups[k]["count"] += 1
    return list(groups.values()), steps


def build_steps(tree, samples, sample_steps, base_format, stop_ids):
    """One record per step (a prefix of a reply): the exact tree's steps, plus the steps samples went through
    that the tree did not follow (below --min-prob). A step's id is the id of the token row that leads to it.
    Each sample is walked token by token; every token it took is marked with the number of samples taking it.
    Returns (steps by id, sample paths, drift), drift = largest |log P| difference between a sample's logged
    step and the tree's step for the same prefix."""
    steps, roots = {}, {}
    for n in tree:
        st = {"id": n["id"], "parent": n["parent"], "cat": n["cat"], "pidx": n["pidx"], "p": math.exp(n["logp"]),
              "root": n["parent"] is None, "entries": []}
        for rank, (tid, text, lpc, end, child, lead) in enumerate(n["next"]):
            st["entries"].append({"tid": tid, "token": text, "cond": math.exp(lpc), "kind": KIND[end],
                                  "id": child if child is not None else f"{n['id']}.{rank}",
                                  "lead": tuple(lead) if lead and lead[0] is not None else None, "n": 0})
        steps[n["id"]] = st
        if st["root"]:
            roots[(n["cat"], n["pidx"])] = n["id"]

    def make_step(sid, parent, cat, pidx, p, logged):
        st = {"id": sid, "parent": parent, "cat": cat, "pidx": pidx, "p": p, "root": False, "sampled": True,
              "entries": [{"tid": tid, "token": text, "cond": math.exp(lp), "kind": "cut", "id": f"{sid}.{rank}",
                           "lead": None, "n": 0} for rank, (tid, text, lp, _) in enumerate(logged["next"])]}
        steps[sid] = st
        return st

    paths, drift, through = [], 0.0, defaultdict(lambda: defaultdict(int))  # entry id -> {answer: samples}
    for smp in sorted(samples, key=lambda r: -r["count"]):
        ids, cur = smp["token_ids"], roots.get((smp["cat"], smp["pidx"]))
        if cur is None:
            continue
        label = smp["answer"] if smp["valid"] else f"[invalid] {smp['answer'] or '<empty>'}"
        text, last, ended = "", None, None
        for j, t in enumerate(ids):
            st, logged = steps[cur], sample_steps.get(smp["steps"][j])
            e = next((e for e in st["entries"] if e["tid"] == t), None)
            lp = next((x[2] for x in logged["next"] if x[0] == t), None) if logged else None
            if e is None:  # sampled a token outside this step's top-k
                e = {"tid": t, "token": next((x[1] for x in logged["next"] if x[0] == t), ""), "cond": math.exp(lp),
                     "kind": "cut", "id": f"{cur}.t{t}", "lead": None, "n": 0}
                st["entries"].append(e)
            elif lp is not None and not st.get("sampled"):
                drift = max(drift, abs(math.log(max(e["cond"], 1e-300)) - lp))
            e["n"] += smp["count"]
            through[e["id"]][label] += smp["count"]
            text += e["token"]
            last = e
            if e["kind"] in ("end", "stop"):  # the tree ends the reply here too
                ended = "end of turn" if e["kind"] == "end" else "line break" if "\n" in e["token"] else "32-token cap"
                break
            if t in stop_ids:
                e["kind"], ended = "end", "end of turn"
                break
            if "\n" in e["token"] and (base_format or first_line_done(text)):
                e["kind"], ended = "stop", "line break"
                break
            if j == len(ids) - 1:
                e["kind"], ended = "stop", "32-token cap" if smp["finish"] == "length" else "stop"
                break
            if e["kind"] != "token" or e["id"] not in steps:  # leaves the exact tree: use the sample's own log
                e["kind"] = "token"
                nxt = sample_steps.get(smp["steps"][j + 1])
                if e["id"] not in steps:
                    make_step(e["id"], cur, st["cat"], st["pidx"], st["p"] * e["cond"], nxt)
            cur = e["id"]
        if last is not None:
            paths.append({"category": smp["cat"], "pidx": smp["pidx"], "reply": smp["raw"], "answer": smp["answer"] or "<empty>",
                          "counts": smp["valid"], "count": smp["count"], "last": str(last["id"]), "ended_by": ended})
    # tokens only samples followed: the answer most of those samples gave
    for st in steps.values():
        for e in st["entries"]:
            if e["lead"] is None and e["id"] in through:
                a, k = max(through[e["id"]].items(), key=lambda x: x[1])
                e["lead"], e["lead_src"] = (a, round(k / sum(through[e["id"]].values()), 4)), "samples"
    return steps, paths, drift


def chooses(st):
    """True where the next token picks the answer: tokens with >= 0.1% lead to two or more answers."""
    return len({e["lead"][0] for e in st["entries"] if e["kind"] != "cut" and e["lead"] and e["cond"] >= 0.001}) > 1


def tree_rows(steps, min_p, full_p):
    """One row per step root (kind root), then every next token the page may show.
      id        id of the step this token leads to; a token nobody followed gets "<parent>.<rank>"
      (category and pidx are on root rows only; other rows inherit them from their root)
      parent    the step it follows
      kind      root | token (the reply goes on) | end (end of turn) | stop (line break or 32-token cap)
                | cut (not followed: below 0.01% overall and not sampled)
      token     token text; cond = P(token | the tokens before it). P(reply so far) is the product of cond
                along the path from the root.
      leads_to  the answer this token mostly leads to, and leads_share its share among answers below it;
                leads_src "samples" where only samples went this way (share among those samples)
      n         how many of the wording's samples took this token (omitted when 0)
      kind rest one row per step: probability on next tokens left out of this table, incl. outside the top-k
    Steps reached with P < min_p are left out unless a sample went through them. The first step of each wording
    and every step where the answer is chosen (reached with P >= full_p) keep all top-k tokens. Other steps keep
    tokens the reply goes on with, sampled tokens, and any other token with P(token | step) >= TREE_MIN_COND;
    the rest is summed in `rest`. tree.jsonl.gz and samples_tree.jsonl.gz have every token of every step."""
    rows = []
    for st in steps.values():
        if st["root"]:
            rows.append({"id": st["id"], "parent": None, "kind": "root", "category": st["cat"], "pidx": st["pidx"],
                         "token": "", "cond": 1.0})
        sampled = any(e["n"] for e in st["entries"])
        if st["p"] < min_p * 0.999 and not sampled:  # open branches have p >= --min-prob; allow for rounding
            continue
        shown, full = 0.0, st["root"] or (st["p"] >= full_p and chooses(st))
        for e in st["entries"]:
            if not full and e["kind"] != "token" and not e["n"] and e["cond"] < TREE_MIN_COND:
                continue
            row = {"id": e["id"], "parent": st["id"], "kind": e["kind"], "token": e["token"], "cond": float(f"{e['cond']:.4g}")}
            if e["lead"]:
                row["leads_to"], row["leads_share"] = e["lead"]
                if e.get("lead_src"):
                    row["leads_src"] = e["lead_src"]
            if e["n"]:
                row["n"] = e["n"]
            rows.append(row)
            shown += e["cond"]
        # mass on this step's next tokens that the table leaves out (incl. outside the top-k)
        rest = max(0.0, 1.0 - shown)
        if rest > 1e-6:
            rows.append({"id": f"{st['id']}.rest", "parent": st["id"], "kind": "rest", "token": "",
                         "cond": float(f"{rest:.4g}")})
    return rows


def main():
    src, out, *models = sys.argv[1:]
    models = models or [m for m in MODEL_LABELS if os.path.isdir(os.path.join(src, m))]
    cats = yaml.safe_load(open(os.path.join(HERE, "..", "eval", "categories.yaml")))
    os.makedirs(out, exist_ok=True)
    acc = defaultdict(list)
    for m in models:
        summary, leaves, tree = load_model(os.path.join(src, m))
        acc["models"].append({"model": m, "label": MODEL_LABELS.get(m, m), "hf_id": summary["model"],
                              "format": summary["format"], "topk": summary["topk"], "min_prob": summary["min_prob"]})
        for name, rows in zip(["categories", "answers", "prompts", "prompt_answers", "replies"],
                              tables(m, summary, leaves, cats)):
            acc[name] += rows
        samples, sample_steps = load_samples(os.path.join(src, m))
        stop_ids = set(summary.get("stop_ids") or [e[0] for n in tree for e in n["next"] if e[3] == "eos"])
        steps, paths, drift = build_steps(tree, samples, sample_steps, summary["format"] == "base", stop_ids)
        acc["samples"] += [dict(r, model=m, id=f"{m}.{r['category']}.{r['pidx']}.{k}") for k, r in enumerate(paths)]
        print(f"{m}: {len(samples)} distinct samples, {sum(s.get('sampled', False) for s in steps.values())} steps "
              f"outside the exact tree, max log-prob drift sample vs tree {drift:.2e}")
        for min_p, full_p in TREE_LEVELS:
            text = json.dumps(tree_rows(steps, min_p, full_p), ensure_ascii=False, separators=(",", ":"))
            if len(text.encode()) <= TREE_MAX_BYTES:
                break
        with open(os.path.join(out, f"tree_{m}.json"), "w") as f:
            f.write(text)
        acc["models"][-1].update(tree_min_p=min_p, tree_full_p=full_p)
        print(f"{m}: {len(tree)} expanded steps -> tree file {len(text.encode()) / 1e6:.1f} MB "
              f"(steps from P >= {min_p:g}, full lists from P >= {full_p:g})")
    for name, rows in acc.items():
        with open(os.path.join(out, f"{name}.json"), "w") as f:
            json.dump(rows, f, ensure_ascii=False, separators=(",", ":"))
        print(f"{name}.json: {len(rows)} rows")


if __name__ == "__main__":
    main()
