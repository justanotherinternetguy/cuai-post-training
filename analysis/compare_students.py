"""Did the student pick up its teacher's collapse? Compares one student's checkpoints with the base model and the teachers.

Usage: python analysis/compare_students.py --student olmo2_1b --prefs <copy of $OUT_DIR/eval/prefs>
           --gen <copy of $OUT_DIR/eval/gen> --data <copy of $OUT_DIR/data> --out analysis/students
Reads:
  prefs/student_<student>__base__chat            base model in the student chat template (step 0)
  prefs/student_<student>__<run>__step<k>        SFT checkpoints; <run> = <data>_n<n>_s<seed>
  prefs/<teacher>                                teachers (the data sets the runs were trained on)
  gen/<student>__base, gen/<student>__<run>__step<k>, data/<teacher>: heldout.stats.json (next-token entropy)
Writes <out>/<student>.md and <out>/<student>_*.png.

Answer entropy is from the exact tree walk, pooled over the 10 wordings exactly as eval/prefs_eval.py does.
JS distance = sqrt(Jensen-Shannon divergence, base 2): 0 = same distribution, 1 = no answer in common.
"""
import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (validated all-pairs, light)
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.sans-serif": ["Noto Sans", "DejaVu Sans"], "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.major.size": 0, "ytick.major.size": 0, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False,
})
TEACHER = {"qwen25_32b": "Qwen2.5-32B", "qwen3_32b": "Qwen3-32B", "olmo2_32b": "OLMo-2-32B",
           "gemma3_27b": "Gemma-3-27B", "mistral31_24b": "Mistral-3.1-24B", "human": "Human replies"}


def pooled(path):
    """Per category: answer -> probability, pooled over wordings like prefs_eval.py (mean of raw mass, renormalized)."""
    w = defaultdict(lambda: defaultdict(float))
    for line in open(os.path.join(path, "logprobs.jsonl")):
        r = json.loads(line)
        w[(r["cat"], r["pidx"])][(r["answer"], r["valid"])] += r["prob"]
    cats = defaultdict(list)
    for (c, p), d in w.items():
        cats[c].append(d)
    out = {}
    for c, per in cats.items():
        tot = defaultdict(float)
        for d in per:
            for k, v in d.items():
                tot[k] += v / len(per)
        nv = sum(v for (a, ok), v in tot.items() if ok)
        out[c] = {a: v / nv for (a, ok), v in tot.items() if ok} if nv else {}
    return out


def ent(d):
    return -sum(p * math.log2(p) for p in d.values() if p > 0)


def js(p, q):
    m = {a: (p.get(a, 0) + q.get(a, 0)) / 2 for a in set(p) | set(q)}
    kl = lambda x: sum(v * math.log2(v / m[a]) for a, v in x.items() if v > 0)  # noqa: E731
    return math.sqrt(max(0.0, (kl(p) + kl(q)) / 2))


def tok_entropy(path):
    f = os.path.join(path, "heldout.stats.json")
    return json.load(open(f))["mean_entropy_stop"] if os.path.exists(f) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--prefs", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="analysis/students")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    S = args.student

    runs = defaultdict(dict)  # run -> step -> prefs dir
    for d in glob.glob(os.path.join(args.prefs, f"student_{S}__*__step*")):
        m = re.match(rf"student_{S}__(.+)__step(\d+)$", os.path.basename(d))
        if m and os.path.exists(os.path.join(d, "logprobs.jsonl")):
            runs[m.group(1)][int(m.group(2))] = d
    base = os.path.join(args.prefs, f"student_{S}__base__chat")
    data_of = {r: r.rsplit("_n", 1)[0] for r in runs}
    teachers = [t for t in dict.fromkeys(data_of.values()) if t != "human" and os.path.isdir(os.path.join(args.prefs, t))]
    order = [r for t in teachers + ["human"] for r in sorted(runs) if data_of[r] == t]

    D = {"base": pooled(base)} if os.path.exists(os.path.join(base, "logprobs.jsonl")) else {}
    for t in teachers:
        D[t] = pooled(os.path.join(args.prefs, t))
    for r in order:
        for st, d in runs[r].items():
            D[(r, st)] = pooled(d)
    cats = list(next(iter(D.values())))
    H = {k: {c: ent(v[c]) for c in cats} for k, v in D.items()}
    meanH = {k: sum(h.values()) / len(h) for k, h in H.items()}
    final = {r: (r, max(runs[r])) for r in order}
    total = {r: max(runs[r]) for r in order}
    TE = {t: tok_entropy(os.path.join(args.data, t)) for t in teachers}
    TE["base"] = tok_entropy(os.path.join(args.gen, f"{S}__base"))
    for r in order:
        for st in runs[r]:
            TE[(r, st)] = tok_entropy(os.path.join(args.gen, f"{S}__{r}__step{st}"))

    def label(k):
        if k == "base":
            return f"{S} base"
        if isinstance(k, tuple):
            return f"SFT on {TEACHER.get(data_of[k[0]], data_of[k[0]])} ({k[1] / total[k[0]]:.0%})"
        return f"Teacher {TEACHER.get(k, k)}"

    L = [f"# Student {S}: does it pick up its teacher's collapse?", "",
         "Answer entropy from the exact tree walk (bits; lower = more collapsed), 10 wordings pooled. "
         "Next-token entropy: mean over the replies to 500 held-out prompts (top-20 lower bound). "
         "All students run in `configs/student_chat.jinja`.", "",
         "## Mean over the 10 questions", "",
         "| Model | Answer entropy | Next-token entropy |", "| --- | --- | --- |"]
    rows = (["base"] if "base" in D else []) + [(r, st) for r in order for st in sorted(runs[r])] + teachers
    for k in rows:
        te = TE.get(k)
        L.append(f"| {label(k)} | {meanH[k]:.2f} | {'-' if te is None else f'{te:.2f}'} |")
    L += ["", "## Per question (final checkpoints)", "",
          "| Question | " + " | ".join(label(k) for k in (["base"] if "base" in D else []) + [final[r] for r in order] + teachers) + " |",
          "| --- |" + " --- |" * ((1 if "base" in D else 0) + len(order) + len(teachers))]
    for c in cats:
        cells = []
        for k in (["base"] if "base" in D else []) + [final[r] for r in order] + teachers:
            top = max(D[k][c].items(), key=lambda x: x[1]) if D[k][c] else ("-", 0)
            cells.append(f"{H[k][c]:.2f} ({top[0]} {top[1]:.0%})")
        L.append(f"| `{c}` | " + " | ".join(cells) + " |")
    L += ["", "## JS distance to each teacher (mean over questions; lower = closer)", "",
          "| Model | " + " | ".join(TEACHER.get(t, t) for t in teachers) + " |", "| --- |" + " --- |" * len(teachers)]
    for k in (["base"] if "base" in D else []) + [final[r] for r in order]:
        L.append(f"| {label(k)} | " + " | ".join(f"{sum(js(D[k][c], D[t][c]) for c in cats) / len(cats):.3f}"
                                                for t in teachers) + " |")

    # Figure 1: answer entropy and next-token entropy over training
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.78, bottom=0.12, wspace=0.22)
    color = {t: SERIES[i % 3] for i, t in enumerate(teachers + ["human"])}
    for ax, vals, ylab in ((axes[0], meanH, "Mean answer entropy (bits)"), (axes[1], TE, "Next-token entropy (bits)")):
        for r in order:
            xs = [0.0] + [st / total[r] for st in sorted(runs[r])]
            ys = [vals.get("base")] + [vals.get((r, st)) for st in sorted(runs[r])]
            pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color[data_of[r]], lw=2, marker="o", ms=6,
                    markeredgecolor=SURF, markeredgewidth=1.5)
        for t in teachers:
            if vals.get(t) is not None:
                ax.axhline(vals[t], color=color[t], lw=1, ls=(0, (4, 3)))
                ax.text(1.0, vals[t], f" {TEACHER[t]} (teacher)", va="bottom", ha="right", fontsize=8, color=INK2)
        ax.set_xlim(-0.03, 1.03)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["base", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("Training progress")
        ax.set_title(ylab, fontsize=10)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.legend(handles=[Line2D([], [], color=color[t], lw=2, marker="o", label=f"SFT on {TEACHER.get(t, t)}")
                        for t in teachers + ["human"] if any(data_of[r] == t for r in order)]
               + [Line2D([], [], color=MUTED, lw=1, ls=(0, (4, 3)), label="Teacher level")],
               loc="upper left", bbox_to_anchor=(0.06, 0.88), ncol=4, fontsize=9)
    fig.text(0.012, 0.975, f"{S}: entropy during fine-tuning", fontsize=14, fontweight="bold", va="top")
    fig.text(0.012, 0.915, "Left: the 10 preference questions (exact tree walk). Right: replies to 500 held-out prompts.",
             fontsize=10, color=INK2, va="top")
    p1 = os.path.join(args.out, f"{S}_entropy_over_training.png")
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    # Figure 2: per-question entropy, final checkpoints vs base and teachers
    keys = (["base"] if "base" in D else []) + [final[r] for r in order]
    fig, ax = plt.subplots(figsize=(13, 5.2))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.78, bottom=0.16)
    w = 0.8 / len(keys)
    for i, k in enumerate(keys):
        col = "#9a9890" if k == "base" else color[data_of[k[0]]]
        ax.bar([j + (i - (len(keys) - 1) / 2) * w for j in range(len(cats))], [H[k][c] for c in cats], width=w * 0.9,
               color=col, label="Base model" if k == "base" else f"SFT on {TEACHER.get(data_of[k[0]])}")
    for t in teachers:
        ax.scatter(range(len(cats)), [H[t][c] for c in cats], marker="_", s=600, color=color[t], linewidths=2.2, zorder=3,
                   label=f"Teacher {TEACHER[t]}")
    ax.set_xticks(range(len(cats)), [c.replace("_", " ") for c in cats], fontsize=9)
    ax.set_ylabel("Answer entropy (bits)")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.14), ncol=6, fontsize=9)
    fig.text(0.012, 0.975, f"{S}: answer entropy per question after fine-tuning", fontsize=14, fontweight="bold", va="top")
    fig.text(0.012, 0.915, "Bars: base model and final checkpoints. Dashes: the teachers.", fontsize=10, color=INK2, va="top")
    p2 = os.path.join(args.out, f"{S}_entropy_per_question.png")
    fig.savefig(p2, dpi=160)
    plt.close(fig)

    L += ["", f"![]({os.path.basename(p1)})", "", f"![]({os.path.basename(p2)})", ""]
    open(os.path.join(args.out, f"{S}.md"), "w").write("\n".join(L))
    print("\n".join(L[:len(rows) + 8]))
    print("wrote", os.path.join(args.out, f"{S}.md"), p1, p2)


if __name__ == "__main__":
    main()
