"""Answer-token distributions: the model's top-k next tokens at the point where it writes its answer.

The logprob tree walk (eval/prefs_eval.py) gives the teachers' exact answer distributions, but it does not work
for SFT students: they answer in sentences ("If I could visit any city, it would be Paris"), so the tree branches
too much (40-90% of the probability is pruned) and its first-line parser misreads the answers. This script
uses the sample-mode logs instead. Every sample logged the model's top-100 next tokens at every step
(samples_tree.jsonl.gz). For each sample:
  1. eval/extract.py reads the answer from the reply (invalid replies are skipped);
  2. the answer step is found: the first prefix of the reply from which extract() returns that answer and keeps
     returning it, then back to the token where the answer's word starts ("**Paris**" -> "Paris", "New York"
     -> " New");
  3. the logged top-100 distribution at that step is read.
Averaging step 3 over samples gives P(answer token | the reply has reached its answer), with the sampled
lead-ins integrated out. It is a lower-noise estimate than counting sampled answers, because each sample
contributes a whole distribution instead of one draw. Teachers get the same treatment, so students and
teachers compare directly; teachers mostly answer at the first token.

Tokens are merged by their text without spaces, markdown or case (" Paris", "Paris", " paris"), and each is
labeled with the answer it leads to: the majority answer of the samples that took it, else the one known
answer (from any model's samples for that question) the token text starts.

Usage (CPU; needs only the tokenizers):
  python analysis/answer_tokens.py --prefs $OUT_DIR/eval/prefs --out $OUT_DIR/eval/answer_tokens [name ...]
Names default to every folder in --prefs with samples_tree.jsonl.gz. Writes <out>/<name>.json:
  model, n_samples, n_valid, n_located and, per question (`cats`) and per wording (`prompts`):
    n_located, coverage (located / valid), mean_step (token index of the answer), top1_p,
    entropy_bits (over the merged tokens, the mass outside every top-100 as one "other" bucket),
    answers: [[answer, P], ...] (merged tokens summed by label; "(not an answer)": tokens with no answer label,
      e.g. "**" or "the"; "(outside top-k)": mass outside the logged top-100),
    tokens: [[key, text, P, answer, n_taken], ...]
"""
import argparse
import gzip
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "eval"))
from extract import extract  # noqa: E402
from normalize import NUM_WORDS  # noqa: E402

TOP_TOKENS = 50      # merged tokens kept per question
TOP_TOKENS_PROMPT = 20


def token_key(text):
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).lower()


def surfaces(ans, cat, cname):
    """Regexes for how answer `ans` can be written in a reply, most specific first."""
    kind = cat["kind"]
    if kind == "int":
        n = int(ans)
        words = [w for w, v in NUM_WORDS.items() if v == n]
        return [re.compile(r"(?<![\d.])" + ans + r"(?![\d])")] + [re.compile(r"\b" + w + r"\b", re.I) for w in words]
    if kind == "letter":
        return [re.compile(r"(?<![A-Za-z])" + ans + r"(?![A-Za-z])"), re.compile(r"(?<![A-Za-z])" + ans.lower() + r"(?![A-Za-z])")]
    if kind == "choice":
        stem = ans.rstrip("s")
        return [re.compile(r"\b" + re.escape(stem) + r"s?\b", re.I)]
    forms = [ans] + [k for k, v in (cat.get("synonyms") or {}).items() if v == ans and k != ans]
    out = []
    for f in forms:
        words = [re.escape(w) for w in f.split()]
        out.append(re.compile(r"(?<!\w)" + r"[\W_]+".join(words) + r"(?!\w)", re.I))
    return out


def answer_token(ids, tok, cat, cname, ans):
    """Index of the token where the answer starts, or None. See the module docstring, step 2."""
    texts = [tok.decode(ids[:j + 1], skip_special_tokens=True) for j in range(len(ids))]
    got = [extract(t, cat, cname)[0] for t in texts]
    stable = None
    for j in range(len(ids) - 1, -1, -1):  # first j from which every longer prefix gives `ans`
        if got[j] != ans:
            break
        stable = j
    if stable is None:
        return None
    text = texts[stable]
    start = None
    for rx in surfaces(ans, cat, cname):
        ms = list(rx.finditer(text))
        if ms:
            start = ms[-1].start()  # the occurrence nearest the end of the prefix: the one that completed it
            break
    if start is None:
        return stable
    for j, t in enumerate(texts):
        if len(t) > start:
            return j
    return stable


def scan_samples(path, tok, cats):
    """Per sample: (cat, pidx, answer, step id of the answer token, token id taken there), only located ones."""
    out, n, nv = [], 0, 0
    for line in open(path):
        r = json.loads(line)
        n += 1
        cat = cats[r["cat"]]
        ans, ok = extract(r["raw"], cat, r["cat"])
        if not ok:
            continue
        nv += 1
        ids = r["token_ids"]
        j = answer_token(ids, tok, cat, r["cat"], ans)
        if j is None or j >= len(r["steps"]):
            out.append((r["cat"], r["pidx"], ans, None, None, None))
            continue
        out.append((r["cat"], r["pidx"], ans, r["steps"][j], ids[j], j))
    return out, n, nv


def read_steps(path, wanted):
    """{step id: next list} for the wanted step ids, streaming the (large) samples_tree.jsonl.gz."""
    got = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            i = int(line[7:line.index(",")])  # lines start with {"id": <n>,
            if i in wanted:
                got[i] = json.loads(line)["next"]
    return got


def entropy(ps):
    return max(0.0, -sum(p * math.log2(p) for p in ps if p > 0))


def summarize(samples, nexts, known):
    """samples: located rows of one question (or wording). Returns the stats dict."""
    loc = [s for s in samples if s[3] is not None]
    if not loc:
        return {"n_located": 0, "coverage": 0.0 if samples else None}
    mass = defaultdict(float)
    text = defaultdict(Counter)
    taken = defaultdict(Counter)  # key -> answers of samples that took it
    for cat, pidx, ans, step, tid, j in loc:
        for t, tx, lp, _ in nexts[step]:
            k = token_key(tx)
            mass[k] += math.exp(lp) / len(loc)
            text[k][tx] += 1
            if t == tid:
                taken[k][ans] += 1
    other = max(0.0, 1.0 - sum(mass.values()))

    def label(k):
        if not k:  # punctuation, markdown, a partial UTF-8 character: not the start of an answer
            return None
        if taken[k]:
            return taken[k].most_common(1)[0][0]
        hits = [a for a in known if token_key(a).startswith(k)]
        return hits[0] if len(hits) == 1 else (k if k in known else None)

    ranked = sorted(mass.items(), key=lambda x: -x[1])
    answers = defaultdict(float)
    for k, p in ranked:
        answers[label(k) or "(not an answer)"] += p
    if other:
        answers["(outside top-k)"] += other
    return {"n_located": len(loc), "coverage": round(len(loc) / len(samples), 4),
            "mean_step": round(sum(s[5] for s in loc) / len(loc), 2),
            "top1_p": round(ranked[0][1], 4), "entropy_bits": round(entropy([p for _, p in ranked] + [other]), 4),
            "other": round(other, 4),
            "answers": [[a, round(p, 5)] for a, p in sorted(answers.items(), key=lambda x: -x[1]) if p >= 1e-4],
            "tokens": [[k, text[k].most_common(1)[0][0], round(p, 5), label(k), sum(taken[k].values())]
                       for k, p in ranked[:TOP_TOKENS]]}


def run(name, prefs, out, known, tokenizer=None):
    from transformers import AutoTokenizer
    d = os.path.join(prefs, name)
    summary = json.load(open(os.path.join(d, "summary.json")))
    tok = AutoTokenizer.from_pretrained(tokenizer or summary["model"])
    cats = yaml.safe_load(open(os.path.join(ROOT, "eval", "categories.yaml")))
    rows, n, nv = scan_samples(os.path.join(d, "samples.jsonl"), tok, cats)
    nexts = read_steps(os.path.join(d, "samples_tree.jsonl.gz"), {r[3] for r in rows if r[3] is not None})
    res = {"name": name, "model": summary["model"], "format": summary.get("format"), "n_samples": n, "n_valid": nv,
           "n_located": sum(r[3] is not None for r in rows), "cats": {}, "prompts": {}}
    for c in cats:
        rc = [r for r in rows if r[0] == c]
        res["cats"][c] = summarize(rc, nexts, known.get(c, set()))
        per = {}
        for i in sorted({r[1] for r in rc}):
            s = summarize([r for r in rc if r[1] == i], nexts, known.get(c, set()))
            s["tokens"] = s.get("tokens", [])[:TOP_TOKENS_PROMPT]
            s.pop("answers", None)
            per[i] = s
        res["prompts"][c] = per
    json.dump(res, open(os.path.join(out, name + ".json"), "w"), ensure_ascii=False)
    return name, n, nv, res["n_located"]


def known_answers(prefs, names, cats):
    """Every valid answer any of the models gave, per question (labels for tokens no sample took)."""
    known = defaultdict(set)
    for name in names:
        for line in open(os.path.join(prefs, name, "samples.jsonl")):
            r = json.loads(line)
            a, ok = extract(r["raw"], cats[r["cat"]], r["cat"])
            if ok:
                known[r["cat"]].add(a)
    return dict(known)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tokenizer", help="tokenizer for every model (default: each model's own, from summary.json)")
    ap.add_argument("--skip-done", action="store_true", help="skip names whose output already exists")
    ap.add_argument("names", nargs="*")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    names = args.names or sorted(n for n in os.listdir(args.prefs)
                                 if os.path.exists(os.path.join(args.prefs, n, "samples_tree.jsonl.gz")))
    cats = yaml.safe_load(open(os.path.join(ROOT, "eval", "categories.yaml")))
    known = known_answers(args.prefs, names, cats)
    if args.skip_done:
        names = [n for n in names if not os.path.exists(os.path.join(args.out, n + ".json"))]
    print(f"{len(names)} models, known answers per question: { {c: len(v) for c, v in known.items()} }", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        for name, n, nv, nl in ex.map(run, names, [args.prefs] * len(names), [args.out] * len(names),
                                      [known] * len(names), [args.tokenizer] * len(names)):
            print(f"{name}: {n} samples, {nv} valid, {nl} answer tokens located ({nl / max(nv, 1):.1%})", flush=True)
    print("ANSWER_TOKENS_OK")


if __name__ == "__main__":
    main()
