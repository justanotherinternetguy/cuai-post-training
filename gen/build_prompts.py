"""Build the IFT prompt pool: human-written single-turn English prompts with their human replies.

Sources: databricks/databricks-dolly-15k (instruction + optional context) and OpenAssistant/oasst1
(first user turn of each English conversation; reply = the top-ranked English assistant answer).
Every teacher answers the same prompts (gen/generate.py); the human replies are the control set.

Removed:
  - "pick one" / preference prompts (checked on the request itself, not Dolly's context passage) like the eval questions (favorite X, pick a number, name a city,
    coin flips, ...) and anything close to an eval wording (eval/categories.yaml), so the student never
    sees the eval questions or their kind;
  - prompts in OLMo-2-32B's own SFT mix (allenai/tulu-3-sft-olmo-2-mixture-0225), so the OLMo teacher
    is not replaying its training data;
  - duplicates, very long prompts or replies (prompt > 3000 chars, reply > 4000 chars), empty replies.

Usage: python gen/build_prompts.py [--heldout 1000] [--seed 0]
Writes to $OUT_DIR/data/:
  prompts/train.jsonl, prompts/heldout.jsonl   {id, source, category, prompt}
  human/train.jsonl, human/heldout.jsonl       {id, prompt, reply, finish: "stop"}
  prompts/removed.jsonl                        every removed prompt with the reason (for review)
  prompts/stats.json
Rows are shuffled once (seed); a run that needs N prompts takes the first N of train.jsonl.
"""
import argparse
import json
import os
import random
import re
from collections import Counter

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.environ.get("OUT_DIR", "."), "data")

# Always removed: these words mark a preference or random-pick question.
ALWAYS = re.compile(r"\b(favou?rites?|random(ly)?|flip (a|the) coin|coin (flip|toss)|toss(ing)? a coin|"
                    r"heads or tails|pick a number|number between|comes? to mind|think of an?)\b", re.I)
# Removed when both appear: a pick/opinion verb and one of the eval topics.
PICK = re.compile(r"\b(pick|choose|chose|select|suggest|recommend|prefer|best|would you|if you could|"
                  r"what is your|what's your|which is your|name (a|an|one|any))\b", re.I)
TOPIC = re.compile(r"\b(numbers?|digits?|integers?|colou?rs?|city|cities|country|countries|destinations?|"
                   r"vacations?|holidays?|animals?|pets?|first names?|baby names?|letters? of the alphabet|"
                   r"letter|fruits?|coins?)\b", re.I)


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def key(s):
    return norm(s)[:300]


def eval_wordings():
    cats = yaml.safe_load(open(os.path.join(HERE, "..", "eval", "categories.yaml")))
    return [set(norm(p).split()) for c in cats.values() for p in c["prompts"]]


def close_to_eval(prompt, wordings):
    words = set(norm(prompt).split())
    if not words or len(words) > 30:
        return False
    return any(len(words & w) / len(words | w) >= 0.5 for w in wordings)


def dolly():
    from datasets import load_dataset
    for i, r in enumerate(load_dataset("databricks/databricks-dolly-15k", split="train")):
        p = r["instruction"].strip()
        if r["context"].strip():
            p = f"{p}\n\n{r['context'].strip()}"
        yield {"id": f"dolly-{i}", "source": "dolly", "category": r["category"], "prompt": p,
               "instruction": r["instruction"].strip(), "reply": r["response"].strip()}


def oasst():
    from datasets import load_dataset
    rows = [r for split in ("train", "validation") for r in load_dataset("OpenAssistant/oasst1", split=split)]
    best = {}
    for r in rows:
        if r["role"] == "assistant" and r["lang"] == "en" and not r["deleted"] and r["rank"] is not None:
            if r["parent_id"] not in best or r["rank"] < best[r["parent_id"]]["rank"]:
                best[r["parent_id"]] = r
    for r in rows:
        if (r["parent_id"] is None and r["role"] == "prompter" and r["lang"] == "en" and not r["deleted"]
                and r["message_id"] in best):
            yield {"id": f"oasst-{r['message_id']}", "source": "oasst", "category": "oasst",
                   "prompt": r["text"].strip(), "instruction": r["text"].strip(),
                   "reply": best[r["message_id"]]["text"].strip()}


def olmo_sft_keys():
    """First user turn of every row in OLMo-2-32B's SFT mix, plus how many rows each source contributes."""
    from datasets import load_dataset
    ds = load_dataset("allenai/tulu-3-sft-olmo-2-mixture-0225", split="train")
    keys, sources = set(), Counter(ds["source"]) if "source" in ds.column_names else Counter()
    for msgs in ds["messages"]:
        for m in msgs:
            if m["role"] == "user":
                keys.add(key(m["content"]))
                break
    return keys, sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    wordings = eval_wordings()
    sft_keys, sft_sources = olmo_sft_keys()
    kept, removed, seen = [], [], set()
    for r in [*dolly(), *oasst()]:
        p, q = r["prompt"], r["instruction"]  # preference checks look at the request, not Dolly's context passage
        reason = None
        if not r["reply"]:
            reason = "empty reply"
        elif len(p) > 3000 or len(r["reply"]) > 4000:
            reason = "too long"
        elif ALWAYS.search(q):
            reason = "preference: " + ALWAYS.search(q).group(0)
        elif PICK.search(q) and TOPIC.search(q):
            reason = f"preference: {PICK.search(q).group(0)} + {TOPIC.search(q).group(0)}"
        elif close_to_eval(q, wordings):
            reason = "close to an eval wording"
        elif key(p) in sft_keys:
            reason = "in OLMo-2 SFT mix"
        elif key(p) in seen:
            reason = "duplicate"
        if reason:
            removed.append({**r, "reason": reason})
            continue
        seen.add(key(p))
        kept.append(r)

    random.Random(args.seed).shuffle(kept)
    split = {"heldout": kept[:args.heldout], "train": kept[args.heldout:]}
    for d in ("prompts", "human"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    for name, rows in split.items():
        with open(os.path.join(OUT, "prompts", f"{name}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps({k: r[k] for k in ("id", "source", "category", "prompt")}) + "\n")
        with open(os.path.join(OUT, "human", f"{name}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps({"id": r["id"], "prompt": r["prompt"], "reply": r["reply"], "finish": "stop"}) + "\n")
    with open(os.path.join(OUT, "prompts", "removed.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in removed)

    reasons = Counter(r["reason"].split(":")[0] for r in removed)
    stats = {"kept": len(kept), "train": len(split["train"]), "heldout": len(split["heldout"]),
             "kept_by_source": Counter(r["source"] for r in kept), "removed": dict(reasons),
             "removed_by_source": Counter(r["source"] for r in removed),
             "olmo_sft_mix_sources": dict(sft_sources.most_common())}
    json.dump(stats, open(os.path.join(OUT, "prompts", "stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))
    rng = random.Random(1)
    print("\nSample of removed preference prompts:")
    for r in rng.sample([r for r in removed if r["reason"].startswith("preference")], 25):
        print(f"  [{r['reason']}] {r['prompt'][:110]!r}")
    print("\nSample of kept prompts:")
    for r in rng.sample(kept, 15):
        print(f"  [{r['source']}/{r['category']}] {r['prompt'][:110]!r}")
    print("BUILD_PROMPTS_OK")


if __name__ == "__main__":
    main()
