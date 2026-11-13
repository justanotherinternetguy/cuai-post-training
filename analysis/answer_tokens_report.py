"""Report on answer-token distributions (analysis/answer_tokens.py output) for one student.

Usage: python analysis/answer_tokens_report.py --tokens <copy of $OUT_DIR/eval/answer_tokens> --student olmo2_1b
           [--out analysis/students]
Writes <out>/<student>_answer_tokens.md and figures <student>_answer_tokens_*.png.

Answer shares are renormalized over answer tokens: the mass at the answer step on tokens that start no answer
("the", "**") and outside the logged top-100 is reported separately as "not an answer".
"""
import argparse
import json
import math
import os
import sys

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rescore_students import COLOR, GRID, INK2, NAME, TEACHERS, clean_axes, title  # noqa: E402  (also sets rcParams)

LABEL = {"number_1_10": "Number 1-10", "number_1_100": "Number 1-100", "vacation_country": "Vacation country",
         "city": "City", "color": "Color", "animal": "Animal", "first_name": "First name", "letter": "Letter",
         "fruit": "Fruit", "coin_flip": "Coin flip"}
NON = ("(not an answer)", "(outside top-k)")
K = 8


def answers(m):
    """Renormalized answer distribution and the non-answer mass at the answer step."""
    if not m or not m.get("answers"):
        return {}, None
    a = {x: p for x, p in m["answers"] if x not in NON}
    tot = sum(a.values())
    return ({x: p / tot for x, p in a.items()} if tot else {}), round(1 - tot, 4)


def entropy(d):
    return -sum(p * math.log2(p) for p in d.values() if p > 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--student", default="olmo2_1b")
    ap.add_argument("--out", default=os.path.join(HERE, "students"))
    args = ap.parse_args()
    S, out = args.student, args.out
    os.makedirs(out, exist_ok=True)

    def load(name):
        f = os.path.join(args.tokens, name + ".json")
        return json.load(open(f)) if os.path.exists(f) else None

    files = os.listdir(args.tokens)
    steps = sorted({int(f.rsplit("step", 1)[1][:-5]) for f in files if f.startswith(f"student_{S}__human_n5000_s0__step")})
    if not steps:
        raise SystemExit(f"no student_{S}__human_n5000_s0__step*.json in {args.tokens}")
    final = steps[-1]
    teachers = [t for t in TEACHERS if load(t) and load(f"student_{S}__{t}_n5000_s0__step{final}")]
    missing = [t for t in TEACHERS if t not in teachers]
    if missing:
        print(f"warning: no teacher or final-checkpoint results for {missing}; left out", file=sys.stderr)
    R = {("teacher", t): load(t) for t in teachers}
    R[("base", "qa")] = load(f"student_{S}__base__qa")
    for d in teachers + ["human"]:
        for st in steps:
            R[(d, st)] = load(f"student_{S}__{d}_n5000_s0__step{st}")
    R = {k: v for k, v in R.items() if v}
    cats = list(LABEL)
    A = {k: {c: answers(v["cats"].get(c)) for c in cats} for k, v in R.items()}  # key -> cat -> (dist, non)

    def lab(k):
        if k[0] == "teacher":
            return f"Teacher {NAME[k[1]]}"
        if k[0] == "base":
            return "Base model (Q/A prompt)"
        return f"Student on {NAME[k[0]]}" + ("" if k[1] == final else f" (step {k[1]})")

    rows = [("base", "qa")] + [(d, final) for d in teachers + ["human"]] + [("teacher", t) for t in teachers]
    rows = [k for k in rows if k in R]
    fav = {t: {c: max(A[("teacher", t)][c][0].items(), key=lambda x: x[1])[0] for c in cats} for t in teachers}

    def fav_share(k, t, c):
        return A[k][c][0].get(fav[t][c], 0.0) if k in A else None

    L = [f"# {S}: answer-token distributions\n",
         "At the step where a reply starts its answer (found with `eval/extract.py`), the model's logged top-100 "
         "next-token distribution, averaged over sampled replies (`analysis/answer_tokens.py`). Tokens are merged "
         "by text and labeled with the answer they lead to; shares are renormalized over answer tokens. "
         "\"Not an answer\" is the mass at that step on tokens that start no answer (\"the\", \"**\") or fall "
         "outside the top-100. Teachers mostly answer at the first token, students a few tokens in.\n",
         "## Summary per model (mean over the 10 questions)\n",
         "| Model | Answer step (token) | Not an answer | Answer-token entropy (bits) | Located / valid replies |",
         "| --- | --- | --- | --- | --- |"]
    for k in rows:
        ms = [R[k]["cats"][c] for c in cats if R[k]["cats"][c].get("n_located")]
        H = [entropy(A[k][c][0]) for c in cats if A[k][c][0]]
        non = [A[k][c][1] for c in cats if A[k][c][1] is not None]
        L.append(f"| {lab(k)} | {sum(m['mean_step'] for m in ms) / len(ms):.1f} | {sum(non) / len(non):.0%} | "
                 f"{sum(H) / len(H):.2f} | {R[k]['n_located']:,} / {R[k]['n_valid']:,} |")

    L += ["\n## Favorite-answer share at the answer token: own teacher vs human control\n",
          "P(answer token = that teacher's top answer), mean over questions. Same measure as the sample-based "
          "favorite share in the rescored report, but from token probabilities.\n",
          "| Teacher | Teacher itself | Student on it | Human control | Base model | Difference (student - control) |",
          "| --- | --- | --- | --- | --- | --- |"]
    for t in teachers:
        mean = lambda k: sum(fav_share(k, t, c) for c in cats) / len(cats)  # noqa: E731
        s, h = mean((t, final)), mean(("human", final))
        b = mean(("base", "qa")) if ("base", "qa") in A else float("nan")
        L.append(f"| {NAME[t]} | {mean(('teacher', t)):.0%} | {s:.1%} | {h:.1%} | {b:.1%} | {100 * (s - h):+.1f} pts |")

    L += ["\n### Per question: student on teacher / human control\n",
          "| Question | " + " | ".join(NAME[t] for t in teachers) + " |", "| --- |" + " --- |" * len(teachers)]
    for c in cats:
        L.append(f"| `{c}` | " + " | ".join(f"“{fav[t][c]}” {fav_share((t, final), t, c):.0%} / "
                                              f"{fav_share(('human', final), t, c):.0%}" for t in teachers) + " |")

    L += ["\n### Over training: student on teacher minus human control (points, mean over questions)\n",
          "| Teacher | " + " | ".join(f"step {s}" for s in steps) + " |", "| --- |" + " --- |" * len(steps)]
    for t in teachers:
        cells = []
        for st in steps:
            if (t, st) in A and ("human", st) in A:
                cells.append(f"{100 * sum(fav_share((t, st), t, c) - fav_share(('human', st), t, c) for c in cats) / len(cats):+.1f}")
            else:
                cells.append("-")
        L.append(f"| {NAME[t]} | " + " | ".join(cells) + " |")

    L += ["\n## Top answer tokens per question (final checkpoints)\n",
          "Each cell: the likeliest answer tokens with their renormalized share; \"not an answer\" mass in brackets.\n"]
    for c in cats:
        L += [f"### {LABEL[c]}\n", "| Model | Top answers | Not an answer |", "| --- | --- | --- |"]
        for k in rows:
            d, non = A[k][c]
            top = ", ".join(f"{a} {p:.0%}" for a, p in sorted(d.items(), key=lambda x: -x[1])[:6])
            L.append(f"| {lab(k)} | {top} | {non:.0%} |" if non is not None else f"| {lab(k)} | - | - |")
        L.append(f"\n![{LABEL[c]}]({S}_answer_tokens_{cats.index(c) + 1:02d}_{c}.png)\n")
    L.insert(5, f"\n![Favorite share at the answer token]({S}_answer_tokens_favorites.png)\n")
    open(os.path.join(out, f"{S}_answer_tokens.md"), "w").write("\n".join(L) + "\n")

    # per-question figure: one row per model, top-K answer tokens (token text, labeled by answer)
    def color(k):
        return COLOR["base"] if k[0] == "base" else COLOR[k[1] if k[0] == "teacher" else k[0]]

    for qi, c in enumerate(cats):
        fig, axes = plt.subplots(len(rows), 1, figsize=(10, 1.05 * len(rows) + 1.2), gridspec_kw={"hspace": 1.1})
        for ax, k in zip(axes, rows):
            m = R[k]["cats"][c]
            toks = [x for x in m.get("tokens", []) if x[3]][:K]  # labeled tokens only
            labeled = sum(p for a, p in A[k][c][0].items()) or 1
            ans_mass = 1 - (A[k][c][1] or 0)
            vals = [x[2] / ans_mass / labeled for x in toks]
            names = [repr(x[1])[1:-1].replace(" ", "·") + (f" → {x[3]}" if x[3].lower() != x[0] else "") for x in toks]
            ax.bar(range(len(vals)), vals, color=color(k), width=0.72)
            for i, v in enumerate(vals):
                ax.text(i, v + 0.03, f"{v:.0%}", ha="center", va="bottom", fontsize=7, color=INK2)
            ax.set_xticks(range(len(vals)), [n if len(n) <= 20 else n[:19] + "…" for n in names], fontsize=7.5)
            ax.set_xlim(-0.6, K - 0.4)
            ax.set_ylim(0, 1.2)
            ax.set_yticks([])
            clean_axes(ax, grid=None)
            ax.spines["left"].set_visible(False)
            non = A[k][c][1]
            ax.set_title(f"{lab(k)}    not an answer: {non:.0%}" if non is not None else lab(k), fontsize=8.5,
                         color=INK2, loc="left", fontweight="normal")
        title(fig, f"{LABEL[c]}: top answer tokens at the answer step",
              f"{S} students (final checkpoints) and teachers. Share of answer-token probability; "
              "token text → the answer it leads to.")
        fig.subplots_adjust(top=1 - 0.9 / (1.05 * len(rows) + 1.2), bottom=0.03, left=0.04, right=0.98)
        fig.savefig(os.path.join(out, f"{S}_answer_tokens_{qi + 1:02d}_{c}.png"), dpi=130)
        plt.close(fig)

    # favorites: per teacher, teacher / student / control / base
    fig, ax = plt.subplots(figsize=(10, 4.2))
    w = 0.2
    kinds = [("teacher", "Teacher itself"), ("student", "Student on it"), ("human", "Human-reply control"),
             ("base", "Base model")]
    for j, (kind, name) in enumerate(kinds):
        ys = []
        for t in teachers:
            k = {"teacher": ("teacher", t), "student": (t, final), "human": ("human", final), "base": ("base", "qa")}[kind]
            ys.append(sum(fav_share(k, t, c) for c in cats) / len(cats) if k in A else 0)
        cols = [COLOR[t] for t in teachers] if kind in ("teacher", "student") else [COLOR[kind]] * len(teachers)
        ax.bar([i + (j - 1.5) * w for i in range(len(teachers))], ys, w * 0.92, color=cols,
               alpha=0.45 if kind == "teacher" else 1.0, label=name)
        for i, y in enumerate(ys):
            ax.text(i + (j - 1.5) * w, y + 0.01, f"{y:.0%}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(teachers)), [NAME[t] for t in teachers])
    ax.set_ylabel("P(teacher's top answer)")
    ax.set_ylim(0, 1)
    ax.yaxis.grid(True, color=GRID)
    clean_axes(ax)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#9a9a9a", alpha=0.45, label="Teacher itself"),
                       Patch(color="#9a9a9a", label="Student on that teacher"),
                       Patch(color=COLOR["human"], label="Human-reply control"),
                       Patch(color=COLOR["base"], label="Base model")], ncol=4, loc="upper right", fontsize=8.5)
    title(fig, "Does the student put more answer-token probability on its teacher's favorites?",
          f"{S}, final checkpoints; mean over the 10 questions of P(answer token = the teacher's top answer)")
    fig.subplots_adjust(top=0.8, bottom=0.1, left=0.08, right=0.98)
    fig.savefig(os.path.join(out, f"{S}_answer_tokens_favorites.png"), dpi=130)
    plt.close(fig)
    print(f"wrote {out}/{S}_answer_tokens.md and {len(cats) + 1} figures")


if __name__ == "__main__":
    main()
