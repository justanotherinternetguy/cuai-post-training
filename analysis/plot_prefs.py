"""Charts for the teacher preference eval. Usage: python analysis/plot_prefs.py <prefs dir> analysis/figures
<prefs dir> holds one folder per model (summary.json, logprobs.jsonl, samples.jsonl) and run_info.json.
Models without a summary.json are skipped. Old PNGs in the output folder are deleted first.
Writes fig*.png plus index.json (file + caption per figure, in order), which make_report.py reads.

Unless a title says "sampling", numbers are from the exact token-tree walk, pooled the same way as
eval/prefs_eval.py: the probability of each (answer, valid) is averaged over the 10 wordings, then
renormalized over valid answers."""
import glob
import json
import math
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SRC, OUT = sys.argv[1:3]
os.makedirs(OUT, exist_ok=True)
for f in glob.glob(os.path.join(OUT, "*.png")):
    os.remove(f)

# Reference palette (dataviz skill, light mode)
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
B600, B450, B250 = "#184f95", "#2a78d6", "#86b6ef"  # ordinal blue steps
ORANGE = "#eb6834"  # categorical slot 2; blue+orange validated (validate_palette.js, light: all checks pass)
REST = "#d9d8d2"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
       "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq", SEQ)

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.sans-serif": ["Noto Sans", "DejaVu Sans"], "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.major.size": 0, "ytick.major.size": 0, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False,
})

M = [("qwen25_32b", "Qwen2.5-32B"), ("qwen3_32b", "Qwen3-32B"), ("olmo2_32b", "OLMo-2-32B"),
     ("gemma3_27b", "Gemma-3-27B"), ("mistral31_24b", "Mistral-3.1-24B")]
M = [(k, n) for k, n in M if os.path.exists(f"{SRC}/{k}/summary.json")]
NAME = dict(M)
D = {k: json.load(open(f"{SRC}/{k}/summary.json")) for k, _ in M}
CATS = list(D[M[0][0]]["logprob"])
LABEL = {"number_1_10": "Number 1-10", "number_1_100": "Number 1-100", "vacation_country": "Vacation country",
         "city": "City", "color": "Color", "animal": "Animal", "first_name": "First name", "letter": "Letter A-Z",
         "fruit": "Fruit", "coin_flip": "Coin flip"}
QUESTION = {"number_1_10": "Pick a number between 1 and 10", "number_1_100": "Pick a number between 1 and 100",
            "vacation_country": "Best country for a holiday", "city": "Favorite city", "color": "Favorite color",
            "animal": "Name an animal", "first_name": "Name a first name", "letter": "Pick a letter A-Z",
            "fruit": "Favorite fruit", "coin_flip": "Flip a coin"}
DOMAIN = {"number_1_10": [str(i) for i in range(1, 11)], "number_1_100": [str(i) for i in range(1, 101)],
          "letter": [chr(c) for c in range(65, 91)]}

# Raw weights per (model, mode, category, wording): {(answer, valid): mass}
W = {}
for k, _ in M:
    for mode, fn, key in (("tree", "logprobs.jsonl", "prob"), ("sample", "samples.jsonl", None)):
        w = defaultdict(lambda: defaultdict(float))
        for line in open(f"{SRC}/{k}/{fn}"):
            r = json.loads(line)
            w[(r["cat"], r["pidx"])][(r["answer"], r["valid"])] += r[key] if key else 1.0
        W[(k, mode)] = w


def wordings(k, c, mode="tree"):
    w = W[(k, mode)]
    return [w[key] for key in sorted(key for key in w if key[0] == c)]


def renorm(weights):
    nv = sum(v for (a, ok), v in weights.items() if ok)
    return {a: v / nv for (a, ok), v in weights.items() if ok} if nv else {}


def pooled(k, c, mode="tree"):
    per = wordings(k, c, mode)
    tot = defaultdict(float)
    for w in per:
        s = sum(w.values())
        for key, v in w.items():
            tot[key] += v / s / len(per) if mode == "sample" else v / len(per)
    return renorm(tot)


def ent(dist):
    return -sum(p * math.log2(p) for p in dist.values() if p > 0)


def ranked(dist):
    return sorted(dist.items(), key=lambda x: -x[1])


P = {(k, c): pooled(k, c) for k, _ in M for c in CATS}
PS = {(k, c): pooled(k, c, "sample") for k, _ in M for c in CATS}
H = {key: ent(d) for key, d in P.items()}
HS = {key: ent(d) for key, d in PS.items()}
HW = {(k, c): sum(ent(renorm(w)) for w in wordings(k, c)) / len(wordings(k, c)) for k, _ in M for c in CATS}
for (k, c), h in H.items():  # same pooling as the eval, so these must match its summary
    assert abs(h - D[k]["logprob"][c]["entropy_bits"]) < 2e-3, (k, c, h)
    assert abs(HW[(k, c)] - D[k]["logprob"][c]["mean_prompt_entropy_bits"]) < 2e-3, (k, c)

INDEX = []


def clean(ax, grid_axis=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis)
        ax.set_axisbelow(True)


def title(fig, main, sub):
    h = fig.get_figheight()
    fig.text(0.012, 1 - 0.18 / h, main, ha="left", va="top", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.012, 1 - 0.5 / h, sub, ha="left", va="top", fontsize=10, color=INK2)


def save(fig, name, caption):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=160)
    plt.close(fig)
    INDEX.append({"file": name, "caption": caption})
    print("wrote", p)


def short(a):
    a = str(a)
    return a.title() if len(a) > 1 else a


def pct(v, nd=0):
    return f"{v:.{nd}%}"


def heat(ax, vals, fmt, vmax, rows, cols):
    """Teacher-column heatmap with a label in every cell; ink flips to white on dark cells."""
    im = ax.pcolormesh(vals, cmap=CMAP, vmin=0, vmax=vmax, edgecolors=SURF, linewidth=3)
    ax.invert_yaxis()
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j + 0.5, i + 0.5, fmt(i, j), ha="center", va="center", fontsize=9.5, linespacing=1.3,
                    color="#ffffff" if vals[i][j] / vmax >= 0.55 else INK)
    ax.set_xticks([j + 0.5 for j in range(len(cols))], cols)
    ax.xaxis.tick_top()
    ax.set_yticks([i + 0.5 for i in range(len(rows))], rows)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def colorbar(fig, im, ax, ticks, labels):
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, ticks=ticks)
    cb.ax.set_yticklabels(labels)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED)


names = [n for _, n in M]
qlabels = [LABEL[c] for c in CATS]

# ---------- 1. Top answer and its share
fig, ax = plt.subplots(figsize=(10, 7.4))
fig.subplots_adjust(left=0.17, right=0.90, top=0.84, bottom=0.03)
vals = [[ranked(P[(k, c)])[0][1] for k, _ in M] for c in CATS]
im = heat(ax, vals, lambda i, j: f"{short(ranked(P[(M[j][0], CATS[i])])[0][0])}\n{vals[i][j]:.0%}", 1, qlabels, names)
colorbar(fig, im, ax, [0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
title(fig, "Every teacher's most likely answer, by question",
      "Probability of the top answer (exact tree walk, averaged over 10 wordings). Darker = more collapsed.")
save(fig, "fig01_top_answer_heatmap.png", "Top answer and its probability for every teacher and question.")

# ---------- 2. Entropy per teacher and question
fig, ax = plt.subplots(figsize=(10, 7.4))
fig.subplots_adjust(left=0.17, right=0.90, top=0.84, bottom=0.03)
vals = [[H[(k, c)] for k, _ in M] for c in CATS]
hmax = math.ceil(max(max(r) for r in vals))


def maxlabel(c):
    return f" / {math.log2(len(DOMAIN[c])):.1f}" if c in DOMAIN else (" / 1.0" if c == "coin_flip" else "")


im = heat(ax, vals, lambda i, j: f"{vals[i][j]:.2f}", hmax, [LABEL[c] + maxlabel(c) for c in CATS], names)
colorbar(fig, im, ax, list(range(hmax + 1)), [str(t) for t in range(hmax + 1)])
fig.text(0.915, 0.86, "bits", fontsize=9, color=MUTED)
title(fig, "Entropy of each teacher's answers, by question",
      "Bits (exact tree walk, 10 wordings pooled). 0 = always the same answer; darker = more varied. "
      "\"/ x\" = the maximum for closed questions.")
save(fig, "fig02_entropy_heatmap.png", "Entropy in bits for every teacher and question.")

# ---------- 3. Mean entropy per teacher: per-wording vs pooled
pool = {k: sum(H[(k, c)] for c in CATS) / len(CATS) for k, _ in M}
perw = {k: sum(HW[(k, c)] for c in CATS) / len(CATS) for k, _ in M}
order = sorted(M, key=lambda m: pool[m[0]])
fig, ax = plt.subplots(figsize=(9, 4.3))
fig.subplots_adjust(left=0.17, right=0.97, top=0.72, bottom=0.14)
for y, (k, n) in enumerate(order):
    ax.plot([perw[k], pool[k]], [y, y], color=AXIS, lw=2, zorder=1, solid_capstyle="round")
    ax.scatter(perw[k], y, s=90, color=B250, edgecolor=SURF, linewidth=2, zorder=2)
    ax.scatter(pool[k], y, s=90, color=B600, edgecolor=SURF, linewidth=2, zorder=3)
    ax.text(pool[k] + 0.09, y, f"{pool[k]:.2f}", va="center", fontsize=9.5, color=INK)
    ax.text(perw[k] - 0.09, y, f"{perw[k]:.2f}", va="center", ha="right", fontsize=9.5, color=INK2)
ax.set_yticks(range(len(order)), [n for _, n in order])
ax.invert_yaxis()
ax.set_xlim(0, math.ceil(max(pool.values()) + 0.5))
ax.set_xlabel("Mean entropy over the 10 questions (bits). Lower = more collapsed.")
clean(ax, "x")
ax.spines["left"].set_visible(False)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=B250, ms=9, label="One wording (mean over the 10)"),
                   Line2D([], [], marker="o", ls="", color=B600, ms=9, label="All 10 wordings pooled")],
          loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9)
title(fig, f"{order[0][1]} is the most collapsed teacher, {order[-1][1]} the least",
      "Asked one wording of a question, every teacher is even more fixed than across all ten (exact tree walk).")
save(fig, "fig03_entropy_by_teacher.png", "Mean entropy per teacher: one wording vs all 10 wordings pooled.")

# ---------- 4. Top-3 answers per question (small multiples of 100% bars)
H4 = 2.9 * len(M)
fig, axes = plt.subplots(5, 2, figsize=(13, H4))
fig.subplots_adjust(left=0.1, right=0.98, top=1 - 1.3 / H4, bottom=0.04, hspace=0.55, wspace=0.18)
seg = [B600, B450, B250]
for ax, c in zip(axes.T.flatten(), CATS):
    for y, (k, n) in enumerate(M):
        x0 = 0
        for r, (a, p) in enumerate(ranked(P[(k, c)])[:3]):
            ax.barh(y, p, left=x0, height=0.7, color=seg[r], edgecolor=SURF, linewidth=2)
            if p >= 0.011 * len(str(a)) + 0.025:  # label only when the whole answer fits
                ax.text(x0 + p / 2, y, short(a), ha="center", va="center", fontsize=8.5,
                        color="#ffffff" if r < 2 else INK)
            x0 += p
        ax.barh(y, 1 - x0, left=x0, height=0.7, color=REST, edgecolor=SURF, linewidth=2)
    ax.set_yticks(range(len(M)), names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"], fontsize=8)
    ax.set_title(LABEL[c], pad=6)
    for s in ax.spines.values():
        s.set_visible(False)
fig.legend(handles=[Patch(color=B600, label="Most likely answer"), Patch(color=B450, label="2nd"),
                    Patch(color=B250, label="3rd"), Patch(color=REST, label="All other answers")],
           loc="upper right", bbox_to_anchor=(0.98, 1 - 0.55 / H4), ncol=4, fontsize=9.5)
title(fig, "Top three answers per teacher and question",
      "Probability over valid answers (exact tree walk). Segments too narrow for a label are in the tables.")
save(fig, "fig04_top3_answers.png", "Top three answers per teacher for each question.")

# ---------- 5-14. One figure per question: the full distribution per teacher
TOPN = 10
for qi, c in enumerate(CATS):
    fname = f"fig{5 + qi:02d}_q_{c}.png"
    if c in DOMAIN:  # closed set: every candidate in its natural order, one row per teacher
        dom = DOMAIN[c]
        hh = 1.15 + 1.55 * len(M)
        fig, axes = plt.subplots(len(M), 1, figsize=(13 if len(dom) > 30 else 10, hh), sharex=True)
        fig.subplots_adjust(left=0.07, right=0.98, top=1 - 1.2 / hh, bottom=0.45 / hh, hspace=0.55)
        ymax = max(max(P[(k, c)].get(a, 0) for a in dom) for k, _ in M)
        tstep = next(t for t in (0.05, 0.1, 0.25, 0.5) if ymax / t <= 4)
        uniform = 1 / len(dom) >= 0.04 * ymax
        for ax, (k, n) in zip(axes, M):
            d = P[(k, c)]
            ys = [d.get(a, 0) for a in dom]
            top = max(range(len(dom)), key=lambda i: ys[i])
            ax.bar(range(len(dom)), ys, width=0.7 if len(dom) < 30 else 0.8,
                   color=[B600 if i == top else B450 for i in range(len(dom))])
            if uniform:
                ax.axhline(1 / len(dom), color=MUTED, lw=0.9, zorder=0)
            ax.text(top, ys[top] + ymax * 0.04, f"{dom[top]}: {pct(ys[top])}", ha="center", va="bottom",
                    fontsize=8.5, color=INK)
            ax.set_ylim(0, ymax * 1.28)
            ax.set_title(n, fontsize=10, pad=2)
            tk = [i * tstep for i in range(int(ymax * 1.28 / tstep) + 1)]
            ax.set_yticks(tk, [pct(t) for t in tk], fontsize=8)
            clean(ax, "y")
            ax.spines["left"].set_visible(False)
        ticks = [i for i in range(len(dom)) if len(dom) <= 30 or (i + 1) % 10 == 0]
        axes[-1].set_xticks(ticks, [dom[i] for i in ticks], fontsize=8)
        if uniform:
            axes[0].text(len(dom) - 0.5, 1 / len(dom) + ymax * 0.03, f"uniform ({pct(1 / len(dom))})",
                         ha="right", va="bottom", fontsize=8, color=MUTED)
        sub = ("Probability of every possible answer (exact tree walk, 10 wordings pooled). All rows share one scale. "
               + ("Gray line: a uniform random choice." if uniform
                  else f"A uniform random choice would give {pct(1 / len(dom))} to each."))
    else:  # open set (and coin flip): top answers as ranked bars, the rest folded into one bar
        n_show = min(TOPN, max(len(P[(k, c)]) for k, _ in M))
        hh = 1.75 + 0.27 * (n_show + 1)
        fig, axes = plt.subplots(1, len(M), figsize=(14, hh), sharex=True)
        fig.subplots_adjust(left=0.07, right=0.99, top=1 - 1.55 / hh, bottom=0.32 / hh, wspace=0.55)
        folded = any(len(P[(k, c)]) > n_show for k, _ in M)
        xmax = max(max(ranked(P[(k, c)])[0][1], 1 - sum(p for _, p in ranked(P[(k, c)])[:n_show])) for k, _ in M)
        for ax, (k, n) in zip(axes, M):
            r = ranked(P[(k, c)])
            show = r[:n_show]
            rest = 1 - sum(p for _, p in show)
            labels = [short(a) for a, _ in show] + ([f"{len(r) - len(show):,} others"] if len(r) > len(show) else [])
            vals = [p for _, p in show] + ([rest] if len(r) > len(show) else [])
            colors = [B600] + [B450] * (len(show) - 1) + ([REST] if len(r) > len(show) else [])
            ax.barh(range(len(vals)), vals, height=0.62, color=colors)
            for y, v in enumerate(vals):
                ax.text(v + xmax * 0.03, y, pct(v) if v >= 0.005 else "<1%", va="center", fontsize=8, color=INK2)
            ax.set_yticks(range(len(vals)), labels, fontsize=8.5)
            ax.set_ylim(n_show + 0.5, -0.6)
            ax.set_xlim(0, xmax * 1.3)
            ax.set_xticks([])
            ax.set_title(n, fontsize=10, pad=18 if folded else 4)
            if folded:
                ax.text(0, 1.0, f"{len(r):,} distinct answers", transform=ax.transAxes, ha="left", va="bottom",
                        fontsize=8.5, color=INK2)
            for s in ax.spines.values():
                s.set_visible(False)
        sub = ((f"Top {n_show} answers per teacher; the rest are summed in the gray bar. " if folded
                else "Every answer per teacher. ") + "Exact tree walk, 10 wordings pooled; all panels share one scale.")
    tops = {ranked(P[(k, c)])[0][0] for k, _ in M}
    head = f"\"{QUESTION[c]}\": " + (f"every teacher's top answer is {short(tops.pop())}" if len(tops) == 1
                                     else f"{len(tops)} different top answers")
    title(fig, head, sub)
    save(fig, fname, f"Full answer distribution per teacher: {LABEL[c].lower()}.")

# ---------- 15. Per-wording top answer and its share
H15 = 1.8 + 2.35 * 5
fig, axes = plt.subplots(5, 2, figsize=(13, H15), sharex=True)
fig.subplots_adjust(left=0.1, right=0.98, top=1 - 1.75 / H15, bottom=0.04, hspace=0.45, wspace=0.18)
for ax, c in zip(axes.T.flatten(), CATS):
    for y, (k, n) in enumerate(M):
        top = ranked(P[(k, c)])[0]
        for w in wordings(k, c):
            a, p = ranked(renorm(w))[0]
            ax.scatter(p, y, s=46, color=B450 if a == top[0] else ORANGE, edgecolor=SURF, linewidth=1.6, zorder=3)
        ax.plot([top[1], top[1]], [y - 0.32, y + 0.32], color=INK, lw=1.4, zorder=2)
    ax.set_yticks(range(len(M)), names, fontsize=9)
    ax.set_ylim(len(M) - 0.5, -0.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"], fontsize=8)
    ax.set_title(LABEL[c], pad=6)
    clean(ax, "x")
    ax.spines["left"].set_visible(False)
fig.legend(handles=[Line2D([], [], marker="o", ls="", color=B450, ms=7, label="One wording: same top answer as pooled"),
                    Line2D([], [], marker="o", ls="", color=ORANGE, ms=7, label="One wording: a different top answer"),
                    Line2D([], [], color=INK, lw=1.4, label="All 10 wordings pooled")],
           loc="upper left", bbox_to_anchor=(0.01, 1 - 0.85 / H15), ncol=3, fontsize=9.5)
title(fig, "How fixed each teacher is on each single wording",
      "Each dot is one of the 10 wordings: the probability of its most likely answer (exact tree walk).")
save(fig, "fig15_per_wording_top1.png", "Top-answer probability for each of the 10 wordings, per teacher and question.")

# ---------- 16. Sampling vs tree walk
fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.11, wspace=0.25)
pairs = [(k, c) for k, _ in M for c in CATS]
for ax, (xs, ys, lab, unit, hi) in zip(axes, [
        ({p: HS[p] for p in pairs}, {p: H[p] for p in pairs}, "Entropy", "bits", None),
        ({p: ranked(PS[p])[0][1] for p in pairs}, {p: P[p].get(ranked(PS[p])[0][0], 0) for p in pairs},
         "Top-answer share", "", 1.0)]):
    HI = hi or max(max(xs.values()), max(ys.values())) + 0.4
    lo = -0.03 * HI
    ax.plot([0, HI], [0, HI], color=AXIS, lw=1, zorder=1)
    ax.scatter([xs[p] for p in pairs], [ys[p] for p in pairs], s=64, color=B450, edgecolor=SURF, linewidth=2, zorder=2)
    far = max(pairs, key=lambda p: abs(xs[p] - ys[p]))
    f = (lambda v: f"{v:.2f}") if unit else pct
    ax.annotate(f"largest gap: {NAME[far[0]]}, {LABEL[far[1]]}\n({f(xs[far])} vs {f(ys[far])})", (xs[far], ys[far]),
                xytext=(0.55, 0.12), textcoords="axes fraction", fontsize=8.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlim(lo, HI)
    ax.set_ylim(lo, HI)
    ax.set_xlabel(f"{lab} from 1,000 samples" + (f" ({unit})" if unit else ""))
    ax.set_ylabel(f"{lab} from the exact tree walk" + (f" ({unit})" if unit else ""))
    if not unit:
        tk = [0, 0.25, 0.5, 0.75, 1]
        ax.set_xticks(tk, [pct(t) for t in tk])
        ax.set_yticks(tk, [pct(t) for t in tk])
    ax.set_title(lab if unit else "Probability of the most common sampled answer", fontsize=10)
    clean(ax, "both")
gap = sum(abs(HS[p] - H[p]) for p in pairs) / len(pairs)
same = sum(ranked(PS[p])[0][0] == ranked(P[p])[0][0] for p in pairs)
title(fig, "Sampling and the exact tree walk agree",
      f"{len(CATS)} questions x {len(M)} teachers = {len(pairs)} points. Same top answer in {same} of {len(pairs)}; "
      f"mean entropy gap {gap:.2f} bits.")
save(fig, "fig16_sample_vs_tree.png", "Entropy and top-answer share: sampling vs the exact tree walk.")

# ---------- 17. Invalid answers (refusals, empty replies, non-answers)
fig, axes = plt.subplots(1, 2, figsize=(14, 7.4))
fig.subplots_adjust(left=0.1, right=0.93, top=0.82, bottom=0.03, wspace=0.08)
inv = {"sample": [[1 - D[k]["sample"][c]["valid_rate"] for k, _ in M] for c in CATS],
       "tree": [[1 - D[k]["logprob"][c]["valid_rate"] for k, _ in M] for c in CATS]}
vmax = max(max(max(r) for r in v) for v in inv.values())
for ax, (mode, lab) in zip(axes, [("sample", "Sampling (share of 1,000 replies)"),
                                  ("tree", "Tree walk (share of probability)")]):
    v = inv[mode]
    im = heat(ax, v, lambda i, j, v=v: "0" if v[i][j] == 0 else ("<0.1%" if v[i][j] < 0.001 else pct(v[i][j], 1)),
              vmax, qlabels, names)
    ax.set_title(lab, pad=28, fontsize=10)
    ax.tick_params(axis="x", labelsize=8.5)
axes[1].set_yticks([])
colorbar(fig, im, axes, [0, vmax / 2, vmax], ["0%", pct(vmax / 2, 1), pct(vmax, 1)])
worst = max(((k, c) for k, _ in M for c in CATS), key=lambda p: 1 - D[p[0]]["sample"][p[1]]["valid_rate"])
title(fig, f"Invalid answers are rare; {NAME[worst[0]]} has the most",
      "Refusals (\"I don't have personal preferences\"), empty replies and non-answers. They are left out of every "
      "distribution above.")
save(fig, "fig17_invalid_answers.png", "Share of invalid answers per teacher and question, sampling and tree walk.")

# ---------- 18. Probability the tree walk dropped
fig, ax = plt.subplots(figsize=(10, 7.4))
fig.subplots_adjust(left=0.17, right=0.90, top=0.84, bottom=0.03)
vals = [[D[k]["logprob"][c]["pruned_mean"] for k, _ in M] for c in CATS]
mx = [[D[k]["logprob"][c]["pruned_max"] for k, _ in M] for c in CATS]
vmax = max(max(r) for r in vals)
im = heat(ax, vals, lambda i, j: f"{pct(vals[i][j], 1)}\nmax {pct(mx[i][j], 1 if mx[i][j] < 0.1 else 0)}", vmax, qlabels, names)
colorbar(fig, im, ax, [0, vmax / 2, vmax], ["0%", pct(vmax / 2, 1), pct(vmax, 1)])
title(fig, "Probability the tree walk did not follow",
      f"Mean over the 10 wordings (max for the worst wording). Branches below P={D[M[0][0]]['min_prob']} or outside "
      f"the top {D[M[0][0]]['topk']} tokens are dropped.")
save(fig, "fig18_dropped_probability.png", "Probability dropped by the tree walk per teacher and question.")

# ---------- 19. Job time: weight loading vs the eval itself
ri = json.load(open(f"{SRC}/run_info.json")) if os.path.exists(f"{SRC}/run_info.json") else {}
if ri:
    def secs(t):
        h, m, s = (int(x) for x in t.split(":"))
        return 3600 * h + 60 * m + s

    rows = [(NAME[k], ri[k]["load_s"] / 60, (secs(ri[k]["elapsed"]) - ri[k]["load_s"]) / 60, ri[k])
            for k, _ in M if k in ri]
    fig, ax = plt.subplots(figsize=(10, 4.3))
    fig.subplots_adjust(left=0.17, right=0.86, top=0.72, bottom=0.14)
    for y, (n, load, rest, info) in enumerate(rows):
        ax.barh(y, load, height=0.55, color=B450)
        ax.barh(y, rest, left=load, height=0.55, color=ORANGE, edgecolor=SURF, linewidth=2)
        ax.text(load + rest + 0.4, y, f"{load + rest:.0f} min · job {info['job']} · {info['node']}",
                va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, max(r[1] + r[2] for r in rows) * 1.5)
    ax.set_xlabel("Minutes")
    clean(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.legend(handles=[Patch(color=B450, label="Loading weights from /share/cuvl"),
                       Patch(color=ORANGE, label="Eval (sampling + tree walk) and the rest of the job")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9)
    share = sum(r[1] for r in rows) / sum(r[1] + r[2] for r in rows)
    title(fig, f"Loading weights took {share:.0%} of the job time",
          "Slurm elapsed time per teacher (2x RTX A6000, TP=2, gpu partition).")
    save(fig, "fig19_job_time.png", "Job time per teacher: weight loading vs the eval.")

json.dump(INDEX, open(os.path.join(OUT, "index.json"), "w"), indent=1)
print(f"wrote {len(INDEX)} figures + index.json")
