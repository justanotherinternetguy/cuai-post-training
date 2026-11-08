"""Mode collapse in SFT students, with answers re-extracted by eval/extract.py.

The students answer in sentences, so the first-line parser of eval/normalize.py misreads them (e.g. "I can't
pick a number between 1 and 100. Here's one: 37" -> "1"). This script re-scores the sample-mode answers
(samples.jsonl, 100 samples x 10 wordings per question) and asks two questions:
  1. Collapse: is a student trained on a teacher's replies less diverse than the student trained on human replies?
  2. Inheritance: does it pick its own teacher's favorite answers (7, 42, Japan, ...) more than that control does?

Usage: python analysis/rescore_students.py --samples <dir> --gen <copy of $OUT_DIR/eval/gen>
           --data <copy of $OUT_DIR/data> --out analysis/students
<dir> holds <name>.jsonl or <name>/samples.jsonl for each eval name (teachers, student_<student>__...).

Entropy: plug-in entropy (bits) of the valid answers, averaged over 50 random subsets of --n answers, so models
with different valid rates are compared at the same sample size (plug-in entropy grows with n).
Favorite share: the probability that a model gives the teacher's top answer, mean over the 10 questions.
"""
import argparse
import glob
import json
import math
import os
import random
import sys
from collections import Counter

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "eval"))
from extract import extract  # noqa: E402

SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
TEACHERS = ["qwen25_32b", "qwen3_32b", "mistral31_24b", "olmo2_32b"]
NAME = {"qwen25_32b": "Qwen2.5-32B", "qwen3_32b": "Qwen3-32B", "mistral31_24b": "Mistral-3.1-24B",
        "olmo2_32b": "OLMo-2-32B", "human": "Human replies", "base": "Base model"}
# Categorical slots 1-4 (validated with the dataviz validator, light mode); controls in neutral ink.
COLOR = {"qwen25_32b": "#2a78d6", "qwen3_32b": "#eb6834", "mistral31_24b": "#1baf7a", "olmo2_32b": "#eda100",
         "human": INK2, "base": "#a9a8a2"}
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.sans-serif": ["Noto Sans", "DejaVu Sans"], "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.major.size": 0, "ytick.major.size": 0, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False,
})


def load(dirpath, name, cats):
    """cat -> (valid answers, n samples)."""
    f = os.path.join(dirpath, name + ".jsonl")
    if not os.path.exists(f):
        f = os.path.join(dirpath, name, "samples.jsonl")
    if not os.path.exists(f):
        return None
    out = {c: ([], 0) for c in cats}
    for line in open(f):
        r = json.loads(line)
        a, ok = extract(r["raw"], cats[r["cat"]], r["cat"])
        ans, n = out[r["cat"]]
        if ok:
            ans.append(a)
        out[r["cat"]] = (ans, n + 1)
    return out


def ent(counts):
    tot = sum(counts.values())
    return -sum(v / tot * math.log2(v / tot) for v in counts.values() if v)


def sub_ent(ans, n, rng, reps=50):
    if len(ans) <= n:
        return ent(Counter(ans))
    return sum(ent(Counter(rng.sample(ans, n))) for _ in range(reps)) / reps


def dist(ans):
    c = Counter(ans)
    t = sum(c.values()) or 1
    return {k: v / t for k, v in c.items()}


def js(p, q):
    m = {a: (p.get(a, 0) + q.get(a, 0)) / 2 for a in set(p) | set(q)}
    kl = lambda x: sum(v * math.log2(v / m[a]) for a, v in x.items() if v > 0)  # noqa: E731
    return math.sqrt(max(0.0, (kl(p) + kl(q)) / 2))


def tok_entropy(path):
    f = os.path.join(path, "heldout.stats.json")
    return json.load(open(f))["mean_entropy_stop"] if os.path.exists(f) else None


def clean_axes(ax, grid="y"):
    if grid:
        ax.grid(axis=grid)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def title(fig, head, sub):
    fig.text(0.012, 0.975, head, fontsize=14, fontweight="bold", va="top")
    fig.text(0.012, 0.925, sub, fontsize=10, color=INK2, va="top")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", default="olmo2_1b")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "analysis", "students"))
    ap.add_argument("--n", type=int, default=200, help="answers per entropy estimate")
    ap.add_argument("--boot", type=int, default=200, help="bootstrap resamples for the intervals")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    S = args.student
    cats = yaml.safe_load(open(os.path.join(ROOT, "eval", "categories.yaml")))
    C = list(cats)
    rng = random.Random(0)

    runs = {d: f"{d}_n5000_s0" for d in TEACHERS + ["human"]}
    steps = sorted({int(os.path.basename(f).rsplit("step", 1)[1].split(".")[0].split("/")[0])
                    for f in glob.glob(os.path.join(args.samples, f"student_{S}__human_n5000_s0__step*"))})
    final = max(steps)
    M = {}  # key -> cat -> (answers, n)
    for t in TEACHERS:
        M[("teacher", t)] = load(args.samples, t, cats)
    M[("base", "qa")] = load(args.samples, f"student_{S}__base__qa", cats)
    M[("base", "chat")] = load(args.samples, f"student_{S}__base__chat", cats)
    for d, r in runs.items():
        for st in steps:
            M[(d, st)] = load(args.samples, f"student_{S}__{r}__step{st}", cats, )
    M = {k: v for k, v in M.items() if v is not None}

    H = {k: {c: sub_ent(v[c][0], args.n, rng) for c in C} for k, v in M.items()}
    V = {k: {c: len(v[c][0]) / max(1, v[c][1]) for c in C} for k, v in M.items()}
    P = {k: {c: dist(v[c][0]) for c in C} for k, v in M.items()}
    meanH = {k: sum(h.values()) / len(C) for k, h in H.items()}
    meanV = {k: sum(v.values()) / len(C) for k, v in V.items()}
    fav = {t: {c: max(P[("teacher", t)][c].items(), key=lambda x: x[1])[0] for c in C} for t in TEACHERS}
    FS = lambda k, t: sum(P[k][c].get(fav[t][c], 0) for c in C) / len(C)  # noqa: E731
    JS = lambda k, t: sum(js(P[k][c], P[("teacher", t)][c]) for c in C) / len(C)  # noqa: E731
    finals = [(d, final) for d in TEACHERS + ["human"]]
    ctrl = ("human", final)

    # Bootstrap over samples (sampling noise only; one seed per run, so seed-to-seed noise is not included).
    def boot(k, stat):
        vals = []
        for _ in range(args.boot):
            res = {c: [rng.choice(M[k][c][0]) for _ in M[k][c][0]] for c in C}
            vals.append(stat(res))
        vals.sort()
        return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]

    def bH(res):
        return sum(sub_ent(res[c], args.n, rng, reps=5) for c in C) / len(C)

    def bF(t):
        return lambda res: sum(dist(res[c]).get(fav[t][c], 0) for c in C) / len(C)

    # Resampling with replacement duplicates answers and biases plug-in entropy down, so the entropy interval
    # is the point estimate +- 1.96 bootstrap SD rather than bootstrap percentiles.
    def sd_ci(k):
        vals = []
        for _ in range(args.boot // 2):
            res = {c: [rng.choice(M[k][c][0]) for _ in M[k][c][0]] for c in C}
            vals.append(bH(res))
        mu = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))
        return meanH[k] - 1.96 * sd, meanH[k] + 1.96 * sd

    ciH = {k: sd_ci(k) for k in finals}
    ciF = {k: boot(k, bF(k[0] if k[0] != "human" else "qwen25_32b")) for k in finals}  # own teacher
    ciFctrl = {t: boot(ctrl, bF(t)) for t in TEACHERS}

    TE = {("base", "chat"): tok_entropy(os.path.join(args.gen, f"{S}__base"))}
    for t in TEACHERS:
        TE[("teacher", t)] = tok_entropy(os.path.join(args.data, t))
    for d, r in runs.items():
        for st in steps:
            TE[(d, st)] = tok_entropy(os.path.join(args.gen, f"{S}__{r}__step{st}"))

    def lab(k):
        if k[0] == "teacher":
            return f"Teacher {NAME[k[1]]}"
        if k[0] == "base":
            return f"Base model ({'Q/A prompt' if k[1] == 'qa' else 'chat template'})"
        return f"Student on {NAME[k[0]]}" + ("" if k[1] == final else f" (step {k[1]})")

    # ---------------- Figure 1: entropy over training ----------------
    def spread(ys, gap):
        """Label positions near ys, at least `gap` apart, keeping the group centered on the points."""
        order = sorted(range(len(ys)), key=lambda i: ys[i])
        pos = [ys[i] for i in order]
        for _ in range(50):
            for j in range(1, len(pos)):
                if pos[j] - pos[j - 1] < gap:
                    mid = (pos[j] + pos[j - 1]) / 2
                    pos[j - 1], pos[j] = mid - gap / 2, mid + gap / 2
        out = [0.0] * len(ys)
        for i, p_ in zip(order, pos):
            out[i] = p_
        return out

    xs = list(range(len(steps) + 1))
    xl = ["base"] + [f"{st / final:.0%}" for st in steps]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    fig.subplots_adjust(left=0.045, right=0.87, top=0.8, bottom=0.11, wspace=0.42)
    for ax, vals, yl, base_k in ((axes[0], meanH, "Answer entropy (bits, mean of 10 questions)", ("base", "qa")),
                                 (axes[1], TE, "Next-token entropy on held-out prompts (bits)", ("base", "chat"))):
        ends = []
        for d in TEACHERS + ["human"]:
            ys = [vals.get(base_k)] + [vals.get((d, st)) for st in steps]
            pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
            ax.plot(*zip(*pts), color=COLOR[d], lw=2, marker="o", ms=7, markeredgecolor=SURF, markeredgewidth=1.5,
                    zorder=3 if d != "human" else 4)
            ends.append((pts[-1][1], f"Student on {NAME[d]}" if d != "human" else "Student on human replies"))
        tdash = [(vals[("teacher", t)], t) for t in TEACHERS if vals.get(("teacher", t)) is not None]
        for y, t in tdash:
            ax.plot([xs[-1] + 0.25, xs[-1] + 0.6], [y, y], color=COLOR[t], lw=2.5, solid_capstyle="round")
        lo = min([y for y, _ in ends] + [y for y, _ in tdash])
        hi = max([y for y, _ in ends] + [vals[base_k]])
        gap = (hi - lo) * 0.065
        for (y, txt), ly in zip(ends, spread([y for y, _ in ends], gap)):
            ax.annotate(txt, (xs[-1], y), xytext=(xs[-1] + 0.75, ly), va="center", fontsize=8.5, color=INK2,
                        arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8, shrinkA=0, shrinkB=4))
        for (y, t), ly in zip(tdash, spread([y for y, _ in tdash], gap)):
            ax.text(xs[-1] + 0.75, ly, f"Teacher {NAME[t]}", va="center", fontsize=8.5, color=INK2)
        ax.set_xticks(xs, xl)
        ax.set_xlim(-0.3, xs[-1] + 0.65)
        ax.set_xlabel("Training progress (share of 313 steps)")
        ax.set_title(yl, fontsize=10)
        clean_axes(ax)
    title(fig, f"{S}: diversity during fine-tuning (students) vs the teachers (dashes at right)",
          "Left: answers to 'pick one' questions, lower = more collapsed; the base point uses the Q/A prompt. "
          "Right: per-token uncertainty on 500 held-out prompts.")
    p1 = os.path.join(args.out, f"{S}_rescored_entropy_over_training.png")
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    # ---------------- Figure 2: per question, each student vs its teacher and the control ----------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.4), sharey=True)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.84, bottom=0.11, hspace=0.45, wspace=0.06)
    xs = list(range(len(C)))
    for ax, t in zip(axes.flat, TEACHERS):
        ax.plot(xs, [H[ctrl][c] for c in C], ls="none", marker="o", ms=8, color=COLOR["human"],
                markeredgecolor=SURF, markeredgewidth=1.5, label="Student on human replies (control)")
        ax.plot(xs, [H[(t, final)][c] for c in C], ls="none", marker="o", ms=8, color=COLOR[t],
                markeredgecolor=SURF, markeredgewidth=1.5, label=f"Student on {NAME[t]}")
        ax.scatter(xs, [H[("teacher", t)][c] for c in C], marker="_", s=420, linewidths=2.5, color=COLOR[t],
                   label=f"Teacher {NAME[t]}")
        for x, c in zip(xs, C):
            ax.plot([x, x], [H[("teacher", t)][c], H[(t, final)][c]], color=GRID, lw=1.2, zorder=0)
        ax.set_title(f"Trained on {NAME[t]}", fontsize=11)
        ax.set_xticks(xs, [c.replace("_", " ").replace("number ", "num ") for c in C], fontsize=8.5, rotation=30)
        ax.legend(loc="upper left", fontsize=8, ncol=3, bbox_to_anchor=(0, 1.02), handletextpad=0.3,
                  columnspacing=1.0)
        ax.set_ylim(0, max(max(H[(d, final)].values()) for d in TEACHERS + ["human"]) * 1.25)
        clean_axes(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Answer entropy (bits)")
    title(fig, f"{S}: answer entropy per question, final checkpoints",
          "A student that inherited its teacher's collapse would sit below the gray control dot, toward its "
          "teacher's dash.")
    p2 = os.path.join(args.out, f"{S}_rescored_per_question.png")
    fig.savefig(p2, dpi=160)
    plt.close(fig)

    # ---------------- Figure 3: favorite-answer share matrix ----------------
    rows = finals
    mat = [[FS(k, t) for t in TEACHERS] for k in rows]
    tmat = [[FS(("teacher", t2), t) for t in TEACHERS] for t2 in TEACHERS]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1]})
    fig.subplots_adjust(left=0.16, right=0.97, top=0.78, bottom=0.2, wspace=0.55)
    vmax = max(max(r) for r in tmat)
    for ax, m, rl, ttl in ((axs[0], tmat, [f"Teacher {NAME[t]}" for t in TEACHERS], "Teachers"),
                           (axs[1], mat, [f"Student on {NAME[d]}" for d, _ in rows], "Students (final)")):
        ax.imshow(m, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        for i, r in enumerate(m):
            for j, v in enumerate(r):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9,
                        color="#ffffff" if v > vmax * 0.55 else INK)
        for i in range(min(len(m), len(TEACHERS))):
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.6))
        ax.set_xticks(range(len(TEACHERS)), [NAME[t] for t in TEACHERS], fontsize=8.5, rotation=20)
        ax.set_yticks(range(len(rl)), rl, fontsize=8.5)
        ax.set_xlabel("Whose favorite answers")
        ax.set_title(ttl, fontsize=10)
        for s in ax.spines.values():
            s.set_visible(False)
    title(fig, f"{S}: how often a model gives each teacher's favorite answer",
          "Mean over the 10 questions of P(answer = that teacher's top answer). Boxed: own teacher. "
          "Inheritance would make each student's boxed cell beat the human-replies row in that column.")
    p3 = os.path.join(args.out, f"{S}_rescored_favorites.png")
    fig.savefig(p3, dpi=160)
    plt.close(fig)

    # ---------------- Figure 4: the teachers' favorite answers, question by question ----------------
    sel = [c for c in C if len({fav[t][c] for t in TEACHERS}) <= 2][:6]  # questions where teachers agree
    fig, axes = plt.subplots(1, len(sel), figsize=(2.3 * len(sel) + 1, 4.6), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.72, bottom=0.3, wspace=0.12)
    keys = [("base", "qa"), ctrl] + [(t, final) for t in TEACHERS]
    for ax, c in zip(axes, sel):
        a = Counter(fav[t][c] for t in TEACHERS).most_common(1)[0][0]
        vals = [P[k][c].get(a, 0) for k in keys]
        ax.bar(range(len(keys)), vals, color=[COLOR[k[0]] for k in keys], width=0.78)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.015, f"{v:.0%}", ha="center", va="bottom", fontsize=7.5, color=INK2)
        tv = sum(P[("teacher", t)][c].get(a, 0) for t in TEACHERS) / len(TEACHERS)
        ax.axhline(tv, color=INK, lw=1, ls=(0, (4, 3)))
        ax.text(len(keys) - 0.5, tv + 0.02, "teachers", ha="right", va="bottom", fontsize=7.5, color=INK2,
                bbox=dict(boxstyle="square,pad=0.1", fc=SURF, ec="none"))
        ax.set_title(f"{c.replace('_', ' ')}: “{a}”", fontsize=9.5)
        ax.set_xticks(range(len(keys)), ["no SFT", "human"] + [NAME[t].split("-")[0] for t in TEACHERS],
                      rotation=55, fontsize=8)
        ax.set_xlabel("1B student fine-tuned on", fontsize=8)
        ax.set_ylim(0, 1)
        clean_axes(ax)
    axes[0].set_ylabel("Share of answers")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    title(fig, f"{S}: do students adopt the teachers' favorite answers?",
          "Every bar is the OLMo-2 1B student: before fine-tuning (no SFT), fine-tuned on human replies (control), "
          "or fine-tuned on one teacher's replies.\nBar = share of its answers that are the teachers' favorite. "
          "Dashed = mean share among the 4 teachers themselves.")
    p4 = os.path.join(args.out, f"{S}_rescored_teacher_favorites.png")
    fig.savefig(p4, dpi=160)
    plt.close(fig)

    # ---------------- Figure 5: top answer per question (student version of figures/fig01) ----------------
    from matplotlib.colors import LinearSegmentedColormap
    seq = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
           "#1c5cab", "#184f95", "#104281", "#0d366b"]  # same ramp as analysis/plot_prefs.py
    cmap = LinearSegmentedColormap.from_list("seq", seq)
    qlab = {"number_1_10": "Number 1-10", "number_1_100": "Number 1-100", "vacation_country": "Vacation country",
            "city": "City", "color": "Color", "animal": "Animal", "first_name": "First name",
            "letter": "Letter A-Z", "fruit": "Fruit", "coin_flip": "Coin flip"}
    order = ["qwen25_32b", "qwen3_32b", "olmo2_32b", "mistral31_24b"]  # teacher order of fig01
    cols = [("base", "qa"), ctrl] + [(t, final) for t in order]
    heads = ["No SFT", "SFT on\nhuman replies"] + [f"SFT on\n{NAME[t]}" for t in order]
    short = lambda a: a.title() if len(a) > 1 else a  # noqa: E731
    top = {(k, c): max(P[k][c].items(), key=lambda x: x[1]) if P[k][c] else ("-", 0) for k in cols for c in C}
    vals = [[top[(k, c)][1] for k in cols] for c in C]
    fig, ax = plt.subplots(figsize=(11.5, 7.8))
    fig.subplots_adjust(left=0.15, right=0.91, top=0.82, bottom=0.03)
    im = ax.pcolormesh(vals, cmap=cmap, vmin=0, vmax=1, edgecolors=SURF, linewidth=3)
    ax.invert_yaxis()
    for i, c in enumerate(C):
        for j, k in enumerate(cols):
            a, v = top[(k, c)]
            teach = k[0] in TEACHERS and a == fav[k[0]][c]
            ink = "#ffffff" if v >= 0.55 else INK
            ax.text(j + 0.5, i + 0.5, f"{short(a)}\n{v:.0%}", ha="center", va="center",
                    fontsize=9.5, linespacing=1.3, color=ink)
            if teach:  # drawn as a marker: the text font has no star glyph
                ax.plot(j + 0.9, i + 0.2, marker="*", ms=10, color=ink, mec="none")
    ax.set_xticks([j + 0.5 for j in range(len(cols))], heads, fontsize=9, color=INK2)
    ax.xaxis.tick_top()
    ax.set_yticks([i + 0.5 for i in range(len(C))], [qlab.get(c, c) for c in C])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.axvline(2, color=INK2, lw=1.2)  # controls | teacher-trained students
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, ticks=[0, 0.25, 0.5, 0.75, 1])
    cb.ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED)
    title(fig, f"Every {S} student's most likely answer, by question",
          "Share of the top answer among valid answers (1000 samples, answers re-extracted). Darker = more "
          "collapsed. Star in the corner = same top answer as its teacher.\nNo SFT uses the Q/A prompt. Compare with "
          "figures/fig01_top_answer_heatmap.png (the teachers).")
    p5 = os.path.join(args.out, f"{S}_rescored_top_answer_heatmap.png")
    fig.savefig(p5, dpi=160)
    plt.close(fig)

    # ---------------- Report ----------------
    pc = lambda x: f"{x:.0%}"  # noqa: E731
    L = [f"# {S}: mode collapse after SFT on teacher replies (re-scored)", "",
         "Answers re-extracted with `eval/extract.py` (the first-line parser misreads sentence replies). "
         f"Sample mode: 100 samples x 10 wordings per question. Entropy: plug-in, bits, at n = {args.n} valid "
         "answers (mean of 50 subsets). Intervals: 95% bootstrap over samples (one training seed per run, so "
         "seed-to-seed noise is not included).", "",
         "## Final checkpoints", "",
         "| Model | Answer entropy | 95% CI | Valid answers | Own teacher's favorite | Human control, same favorites | Next-token entropy |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for k in [("base", "qa"), ("base", "chat")] + finals:
        own = k[0] if k[0] in TEACHERS else None
        ci = ciH.get(k)
        L.append(f"| {lab(k)} | {meanH[k]:.2f} | {f'{ci[0]:.2f}-{ci[1]:.2f}' if ci else '-'} | {pc(meanV[k])} | "
                 f"{pc(FS(k, own)) if own else '-'} | {pc(FS(ctrl, own)) if own else '-'} | "
                 f"{'-' if TE.get(k) is None else f'{TE[k]:.2f}'} |")
    for t in TEACHERS:
        k = ("teacher", t)
        L.append(f"| {lab(k)} | {meanH[k]:.2f} | | {pc(meanV[k])} | {pc(FS(k, t))} | | "
                 f"{'-' if TE.get(k) is None else f'{TE[k]:.2f}'} |")
    L += ["", "## Favorite-answer share: own teacher vs human control (final checkpoints)", "",
          "| Teacher | Student on it | 95% CI | Human control | 95% CI | Difference |", "| --- | --- | --- | --- | --- | --- |"]
    for t in TEACHERS:
        a, b = FS((t, final), t), FS(ctrl, t)
        L.append(f"| {NAME[t]} | {pc(a)} | {pc(ciF[(t, final)][0])}-{pc(ciF[(t, final)][1])} | {pc(b)} | "
                 f"{pc(ciFctrl[t][0])}-{pc(ciFctrl[t][1])} | {(a - b) * 100:+.1f} pts |")
    L += ["", "## Favorite-answer share per question: student on teacher vs human control", "",
          "Each cell: P(answer = that teacher's top answer) for the student trained on that teacher / for the "
          "human-replies control.", "",
          "| Question | " + " | ".join(NAME[t] for t in TEACHERS) + " |", "| --- |" + " --- |" * len(TEACHERS)]
    for c in C:
        L.append(f"| `{c}` | " + " | ".join(
            f"\u201c{fav[t][c]}\u201d {pc(P[(t, final)][c].get(fav[t][c], 0))} / {pc(P[ctrl][c].get(fav[t][c], 0))}"
            for t in TEACHERS) + " |")
    L += ["", "## JS distance to each teacher (mean over questions; 0 = same, 1 = disjoint)", "",
          "| Model | " + " | ".join(NAME[t] for t in TEACHERS) + " |", "| --- |" + " --- |" * len(TEACHERS)]
    for k in [("base", "qa")] + finals:
        L.append(f"| {lab(k)} | " + " | ".join(f"{'**' if k[0] == t else ''}{JS(k, t):.3f}{'**' if k[0] == t else ''}"
                                                for t in TEACHERS) + " |")
    L += ["", "## Per question: entropy (top answer, share)", "",
          "| Question | " + " | ".join(lab(k) for k in [("base", "qa")] + finals + [("teacher", t) for t in TEACHERS]) + " |",
          "| --- |" + " --- |" * (1 + len(finals) + len(TEACHERS))]
    for c in C:
        cells = []
        for k in [("base", "qa")] + finals + [("teacher", t) for t in TEACHERS]:
            top = max(P[k][c].items(), key=lambda x: x[1]) if P[k][c] else ("-", 0)
            cells.append(f"{H[k][c]:.2f} ({top[0]} {top[1]:.0%})")
        L.append(f"| `{c}` | " + " | ".join(cells) + " |")
    L += ["", "## Over training: answer entropy / valid rate", "",
          "| Run | " + " | ".join(f"step {st}" for st in steps) + " |", "| --- |" + " --- |" * len(steps)]
    for d in TEACHERS + ["human"]:
        L.append(f"| {NAME[d]} | " + " | ".join(f"{meanH[(d, st)]:.2f} / {pc(meanV[(d, st)])}" for st in steps) + " |")
    L += ["", f"![]({os.path.basename(p1)})", "", f"![]({os.path.basename(p2)})", "",
          f"![]({os.path.basename(p3)})", "", f"![]({os.path.basename(p4)})", "", f"![]({os.path.basename(p5)})", ""]
    out = os.path.join(args.out, f"{S}_rescored.md")
    open(out, "w").write("\n".join(L))
    print("\n".join(L[:-8]))
    print("wrote", out, p1, p2, p3, p4)


if __name__ == "__main__":
    main()
