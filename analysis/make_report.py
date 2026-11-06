"""Write the teacher preference-eval results report (Markdown) from the eval outputs.

Usage:
  python analysis/make_report.py <prefs dir> <out .md> [--figures <dir, relative to the .md>]
<prefs dir> holds one folder per model (summary.json) and, optionally, run_info.json: {model name: {job, node,
elapsed, load_s, snapshot, start}} for the jobs table. Models without a summary.json are skipped.
Both modes must be in each summary: "sample" (100 replies per wording) and "logprob" (token-tree walk).
"""
import argparse
import datetime
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "eval"))
from normalize import candidates  # noqa: E402

MODELS = [("qwen25_32b", "Qwen2.5-32B"), ("qwen3_32b", "Qwen3-32B"), ("olmo2_32b", "OLMo-2-32B"),
          ("gemma3_27b", "Gemma-3-27B"), ("mistral31_24b", "Mistral-3.1-24B")]
MODES = [("sample", "sample"), ("logprob", "tree")]


def pct(x, nd=0):
    return "–" if x is None else f"{100 * x:.{nd}f}%"


def tops(t, k):
    return ", ".join(f"{a} {pct(p)}" for a, p in t[:k]) or "–"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefs")
    ap.add_argument("out")
    ap.add_argument("--figures", default="figures")
    args = ap.parse_args()

    cats = yaml.safe_load(open(os.path.join(HERE, "..", "eval", "categories.yaml")))
    M = [(k, n) for k, n in MODELS if os.path.exists(os.path.join(args.prefs, k, "summary.json"))]
    S = {k: json.load(open(os.path.join(args.prefs, k, "summary.json"))) for k, _ in M}
    info_path = os.path.join(args.prefs, "run_info.json")
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    C = list(cats)
    names = [n for _, n in M]
    first = S[M[0][0]]
    L = []
    w = L.append

    def get(k, mode, c, f):
        return S[k][mode][c][f]

    def maxbits(c):
        cand = candidates(cats[c])
        return f"{__import__('math').log2(len(cand)):.2f}" if cand else "open"

    dates = sorted(v["start"][:10] for v in info.values() if v.get("start"))
    run_date = dates[0] if dates else "(date not recorded)"
    w("# Teacher preference eval: results")
    w("")
    w(f"Run on {run_date} on the Cornell Unicorn cluster (user `qx235`). Report written "
      f"{datetime.date.today().isoformat()} by `analysis/make_report.py` from the outputs in "
      f"`/share/cuvl/qx235/unicorn/eval/prefs/`. Setup tests and history: `teacher_tests_report.md`.")
    w("")
    w("Every teacher answered every wording of every question in two ways:")
    w("")
    w(f"- **Sampling:** {first['n']} full replies per wording at temperature 1. The answer distribution is the "
      "share of replies giving each answer.")
    w("- **Token-tree walk:** the exact probability of every answer, computed from the model's next-token "
      "probabilities without sampling (see 1.4).")
    w("")
    w("## 1. Experiment")
    w("")
    w("### 1.1 Research question")
    w("")
    w("Does a base model fine-tuned (SFT) on a post-trained teacher's outputs inherit the teacher's preferences, "
      "in particular its mode collapse on open \"pick one\" questions? This stage measures the **teachers** only. "
      "The same eval will run on the student base models (before SFT) and on the SFT students, with the same prompts.")
    w("")
    w("### 1.2 Models and jobs")
    w("")
    rows = []
    for k, n in M:
        i = info.get(k, {})
        rows.append([n, f"`{S[k]['model']}`", f"`{i['snapshot'][:8]}`" if i.get("snapshot") else "–",
                     i.get("job", "–"), i.get("node", "–"),
                     f"{i['load_s']} s" if i.get("load_s") is not None else "–", i.get("elapsed", "–")])
    w(table(["Teacher", "HF id", "Snapshot", "Job", "Node", "Load time", "Job time"], rows))
    w("")
    w("### 1.3 Settings")
    w("")
    w(table(["Item", "Value"], [
        ["GPUs per job", "2x NVIDIA RTX A6000 (48 GB), tensor parallel = 2"],
        ["Partition", "`gpu`, 4 CPUs, 48 GB RAM, 4 h limit, `--requeue`"],
        ["Engine", "vLLM 0.33.0, `max_model_len=2048`, `gpu_memory_utilization=0.90`, `generation_config=\"vllm\"`"],
        ["Prompt format", f"`{first['format']}`: the model's chat template, thinking off, a reply ends at end-of-turn"],
        ["Sampling", f"{first['n']} replies per wording, T=1.0, top-p 1.0, no top-k, max 32 tokens, seed {first['seed']}"],
        ["Token-tree walk", f"top-{first['topk']} next tokens per step, branches below P={first['min_prob']} dropped, "
                            "max depth 32 tokens"],
        ["Code", "`eval/prefs_eval.py`, `eval/normalize.py`, `eval/categories.yaml`, `slurm/eval_prefs.sub`"],
        ["Command", "`sbatch --exclude=<old-driver nodes> --requeue -t 4:00:00 slurm/eval_prefs.sub "
                    "--model <hf id> --name <name> --out-root <dir>` (both modes, the default)"],
        ["Outputs", "`/share/cuvl/qx235/unicorn/eval/prefs/<name>/`: `samples.jsonl`, `samples_tree.jsonl.gz`, "
                    "`logprobs.jsonl`, `tree.jsonl.gz`, `summary.json`; `run_info.json` (jobs)"],
    ]))
    w("")
    w("End-of-turn token per model: " + ", ".join(f"{n} `{S[k]['end_marker']}`" for k, n in M) + ".")
    w("")
    w("### 1.4 Method")
    w("")
    w(f"**Sampling.** Each wording gets {first['n']} replies at T=1, so 1,000 per question and 10,000 per model. "
      "Each reply is normalized to an answer (below) and the shares are counted. At every step of every reply, the "
      "model's top-100 next tokens are also logged (`samples_tree.jsonl.gz`).")
    w("")
    w("**Token-tree walk.** For each wording:")
    w("")
    w(f"1. Start at the prompt with probability 1. Ask the model for its top-{first['topk']} next tokens and their "
      "probabilities.")
    w("2. Each token opens a branch. A branch's probability is the product of the token probabilities along it.")
    w("3. Ask again for every open branch (all open branches go to vLLM as one batch per step). A branch ends at "
      "end-of-turn, when its first line is complete, or after 32 tokens.")
    w(f"4. Branches below P={first['min_prob']} are dropped. The dropped share is reported as *pruned*.")
    w("5. Each finished branch is a full reply. It is normalized like a sample, and replies with the same answer are "
      "added up (\"Blue\" and \"Blue.\" both count as blue).")
    w("")
    w("This gives the distribution that sampling draws from, without sampling noise. Every model call returns the "
      "same probabilities for the same input, so each branch is asked once.")
    w("")
    w("**Pooling.** A question's distribution is the mean over its 10 wordings (each wording weighted equally, like "
      "pooling 10 equal-sized sample sets). Per-wording numbers are reported too.")
    w("")
    w("**Normalization** (`eval/normalize.py`): first line only, lowercase, strip markdown, quotes and edge "
      "punctuation, drop prefixes like \"my answer is\", cut at the first comma, drop leading articles, apply the "
      "category's synonym map. Numbers: digits or number words. Letters: a single letter, optionally \"the letter X\".")
    w("")
    w("**Valid:** closed questions (numbers, letter, coin flip), the answer is in the answer set. Open questions, a "
      "non-empty answer of at most 4 words. Invalid answers (refusals, empty replies) are reported separately and "
      "left out of the distribution.")
    w("")
    w("### 1.5 Metrics")
    w("")
    w(table(["Metric", "Definition"], [
        ["Entropy (H), bits", "Shannon entropy of the pooled answer distribution. 0 = always the same answer. "
                              "Max for closed sets = log2(size of the set)."],
        ["Top-1 share", "Share of valid answers that are the most common answer"],
        ["Per-wording H", "Entropy for each wording on its own, averaged over the 10 wordings. Leaves out the "
                          "variety that comes only from different wordings."],
        ["Valid", "Share of replies (or probability) that are valid"],
        ["Distinct", "Number of different valid answers with any probability"],
        ["Pruned", "Tree walk only: probability dropped because a branch fell below the cutoff, mean (max) over "
                   "the 10 wordings"],
    ]))
    w("")
    w("### 1.6 Questions")
    w("")
    rows = []
    for c in C:
        cand = candidates(cats[c])
        kind = cats[c]["kind"]
        aset = (f"integers {cats[c]['range'][0]}-{cats[c]['range'][1]}" if kind == "int" else
                "A-Z (26)" if kind == "letter" else ", ".join(cand) if cand else "open (any answer of at most 4 words)")
        rows.append([f"`{c}`", kind, aset, cats[c]["instruction"]])
    w(table(["Question", "Kind", "Answer set", "Instruction"], rows))
    w("")
    w("<details><summary>All 100 wordings</summary>")
    w("")
    for c in C:
        w(f"**`{c}`**")
        w("")
        for i, p in enumerate(cats[c]["prompts"]):
            w(f"{i}. {p}")
        w("")
    w("</details>")
    w("")

    # ---------------- results
    w("## 2. Results")
    w("")
    w("### Figures")
    w("")
    w("Made with `analysis/plot_prefs.py`. Unless a caption says otherwise, numbers come from the exact token-tree "
      "walk, with the 10 wordings pooled. Charts with many answers show the top answers plus \"all other answers\"; "
      "exact numbers are in the tables below.")
    w("")
    # index.json is written by plot_prefs.py: one {file, caption} per figure, in order
    figs = json.load(open(os.path.join(os.path.dirname(os.path.abspath(args.out)), args.figures, "index.json")))
    for f, cap in [(x["file"], x["caption"]) for x in figs]:
        w(f"![{cap}]({args.figures}/{f})")
        w("")
        w(f"*{cap}*")
        w("")

    w("### 2.1 Summary across teachers")
    w("")
    for mode, label in MODES:
        w(f"**Top answer and its share** ({label})")
        w("")
        w(table(["Question"] + names,
                [[f"`{c}`"] + [tops(get(k, mode, c, "top"), 1) for k, _ in M] for c in C]))
        w("")
    w("**Top 3 answers** (tree)")
    w("")
    w(table(["Question"] + names, [[f"`{c}`"] + [tops(get(k, "logprob", c, "top"), 3) for k, _ in M] for c in C]))
    w("")
    for mode, label in MODES:
        w(f"**Entropy, bits** ({label}; bold = most collapsed teacher in the row)")
        w("")
        rows = []
        for c in C:
            hs = [get(k, mode, c, "entropy_bits") for k, _ in M]
            lo = min(hs)
            rows.append([f"`{c}`", maxbits(c)] + [f"**{h:.2f}**" if h == lo else f"{h:.2f}" for h in hs])
        mean = {k: sum(get(k, mode, c, "entropy_bits") for c in C) / len(C) for k, _ in M}
        perw = {k: sum(get(k, mode, c, "mean_prompt_entropy_bits") for c in C) / len(C) for k, _ in M}
        lo = min(mean.values())
        rows.append(["**Mean**", ""] + [f"**{mean[k]:.2f}**" if mean[k] == lo else f"{mean[k]:.2f}" for k, _ in M])
        rows.append(["Mean per-wording H", ""] + [f"{perw[k]:.2f}" for k, _ in M])
        w(table(["Question", "Max"] + names, rows))
        w("")
        order = sorted(M, key=lambda m: mean[m[0]])
        w(f"Ranking, most to least collapsed (mean H, {label}): "
          + " < ".join(f"{n} {mean[k]:.2f}" for k, n in order) + ".")
        w("")
    w("**Sampling vs tree.** Cell: H tree / H sample (bits) · top-1 share tree / sample")
    w("")
    w(table(["Question"] + names, [[f"`{c}`"] + [
        f"{get(k, 'logprob', c, 'entropy_bits'):.2f} / {get(k, 'sample', c, 'entropy_bits'):.2f} · "
        f"{pct(get(k, 'logprob', c, 'top1_share'))} / {pct(get(k, 'sample', c, 'top1_share'))}" for k, _ in M]
        for c in C]))
    w("")
    w("**Valid rate** (sample / tree)")
    w("")
    w(table(["Question"] + names, [[f"`{c}`"] + [
        f"{pct(get(k, 'sample', c, 'valid_rate'))} / {pct(get(k, 'logprob', c, 'valid_rate'))}" for k, _ in M]
        for c in C]))
    w("")
    w("**Pruned probability** (tree): mean (max) over the 10 wordings · finished replies")
    w("")
    w(table(["Question"] + names, [[f"`{c}`"] + [
        f"{pct(get(k, 'logprob', c, 'pruned_mean'), 1)} ({pct(get(k, 'logprob', c, 'pruned_max'), 1)}) · "
        f"{get(k, 'logprob', c, 'n_leaves'):,}" for k, _ in M] for c in C]))
    w("")

    w("### 2.2 Full distributions per question")
    w("")
    w("Top 10 valid answers per teacher. *Per-wording top-1* is the most common answer and its share for each of "
      "the 10 wordings, in the order of the list in 1.6.")
    w("")
    for c in C:
        w(f"#### `{c}`")
        w("")
        rows = []
        for k, n in M:
            for j, (mode, label) in enumerate(MODES):
                m = S[k][mode][c]
                rows.append([n if j == 0 else "", label, pct(m["valid_rate"]), m["n_distinct"],
                             f"{m['entropy_bits']:.2f}", f"{m['mean_prompt_entropy_bits']:.2f}", tops(m["top"], 10)])
        w(table(["Teacher", "Mode", "Valid", "Distinct", "H", "Per-wording H", "Top answers"], rows))
        w("")
        rows = []
        for k, n in M:
            for j, (mode, label) in enumerate(MODES):
                pt = S[k][mode][c]["prompt_top1"]
                rows.append([n if j == 0 else "", label,
                             " · ".join(f"{a} {p:.2f}" if a is not None else "–" for a, p in
                                        [(x[0], x[1]) if x else (None, 0) for x in pt])])
        w(table(["Teacher", "Mode", "Per-wording top-1"], rows))
        w("")

    w("### 2.3 Invalid answers")
    w("")
    w("Most common invalid answers in sampling (count out of 1,000), where the valid rate is below 100%.")
    w("")
    rows = []
    for c in C:
        for k, n in M:
            m = S[k]["sample"][c]
            if m["valid_rate"] < 1:
                inv = round((1 - m["valid_rate"]) * m["n"])
                rows.append([f"`{c}`", n, inv, "; ".join(f"{str(a)[:60]} ({cnt})" for a, cnt in m["top_invalid"])])
    w(table(["Question", "Teacher", "Invalid (of 1,000)", "Top invalid answers"], rows) if rows else "None.")
    w("")
    tr = [f"{n} `{c}` {pct(S[k]['sample'][c]['truncated_rate'], 1)}" for c in C for k, n in M
          if S[k]["sample"][c]["truncated_rate"] > 0]
    w("Samples cut off at 32 tokens: " + (", ".join(tr) if tr else "none") + ".")
    w("")

    # ---------------- findings (computed)
    w("## 3. Findings")
    w("")
    meanH = {k: sum(get(k, "logprob", c, "entropy_bits") for c in C) / len(C) for k, _ in M}
    perH = {k: sum(get(k, "logprob", c, "mean_prompt_entropy_bits") for c in C) / len(C) for k, _ in M}
    order = sorted(M, key=lambda m: meanH[m[0]])
    top1 = {c: {k: get(k, "logprob", c, "top")[0][0] for k, _ in M} for c in C}
    shared = [c for c in C if len(set(top1[c].values())) == 1]
    split = [c for c in C if len(set(top1[c].values())) > 1]
    n_find = 1
    if shared:
        w(f"{n_find}. **Shared top answers.** Every teacher's top answer is the same on {len(shared)} of {len(C)} "
          "questions: " + ", ".join(f"{c} ({top1[c][M[0][0]]})" for c in shared) + ".")
        n_find += 1
    w(f"{n_find}. **Most collapsed: {order[0][1]}; least: {order[-1][1]}** (mean tree entropy "
      + ", ".join(f"{n} {meanH[k]:.2f}" for k, n in order) + " bits).")
    n_find += 1
    if split:
        w(f"{n_find}. **Questions where teachers differ** (useful for telling which teacher a student learned from):")
        for c in split:
            groups = {}
            for k, n in M:
                groups.setdefault(top1[c][k], []).append(n)
            w(f"   - `{c}`: " + "; ".join(f"{a} ({', '.join(ns)})" for a, ns in groups.items()))
        n_find += 1
    w(f"{n_find}. **One wording is far more collapsed than all 10 pooled.** Mean per-wording entropy (tree) vs "
      "pooled: " + ", ".join(f"{n} {perH[k]:.2f} vs {meanH[k]:.2f}" for k, n in M) + " bits. Much of the "
      "pooled variety comes from changing the wording, not from the model choosing at random.")
    n_find += 1
    gaps = [(abs(get(k, "logprob", c, "entropy_bits") - get(k, "sample", c, "entropy_bits")), n, c) for k, n in M
            for c in C]
    agree = sum(get(k, "logprob", c, "top")[0][0] == get(k, "sample", c, "top")[0][0] for k, _ in M for c in C)
    big = max(gaps)
    w(f"{n_find}. **Sampling and the tree walk agree.** Same top answer in {agree} of {len(gaps)} teacher-question "
      f"pairs. Mean entropy gap {sum(g for g, *_ in gaps) / len(gaps):.2f} bits; largest {big[0]:.2f} bits "
      f"({big[1]}, `{big[2]}`).")
    w("")

    w("## 4. Caveats")
    w("")
    hi = sorted(((get(k, "logprob", c, "pruned_max"), get(k, "logprob", c, "pruned_mean"), n, c) for k, n in M
                 for c in C), reverse=True)
    hi = [x for x in hi if x[0] >= 0.05]
    if hi:
        w(f"- **Pruned probability.** Where answers are spread out, many branches fall below P={first['min_prob']} "
          "and are dropped, so the tree walk undercounts entropy there. Wordings losing 5% or more: "
          + ", ".join(f"{n} `{c}` max {pct(mx)} (mean {pct(mn, 1)})" for mx, mn, n, c in hi[:12])
          + ". A lower `--min-prob` would recover it at the cost of more model calls.")
    w("- One seed and one temperature (T=1). Sampling has noise of a few percentage points per share; treat "
      "entropy differences under about 0.1 bit as noise. The tree walk has no sampling noise.")
    w("- Thinking is off for every model. Qwen3's preferences with thinking on are not measured.")
    w("- Open questions depend on the synonym maps in `eval/categories.yaml`. Near-duplicates that are not mapped "
      "count as separate answers, which raises the entropy a little.")
    w("- Refusals (\"I don't have a favorite color\") count as invalid and are left out of the distribution; see "
      "2.3 and the valid-rate table.")
    w("")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(L))
    print("wrote", args.out, f"({len(M)} models)")


if __name__ == "__main__":
    main()
