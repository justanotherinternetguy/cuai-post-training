"""Turn interp/collapse_neurons.py output (results.json) into a markdown summary + figures.

Usage: python interp/report.py <results.json> <out.md>
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = json.load(open(sys.argv[1]))
out = sys.argv[2]
figdir = os.path.splitext(out)[0] + "_figs"
os.makedirs(figdir, exist_ok=True)
L = []
p = L.append


def ap(d):
    return f"H={d['H']:.2f} bits, P(top)={d['p_top']:.2f}"


def short(mid):
    """allenai/OLMo-2-0425-1B-SFT -> OLMo-2-0425-1B-SFT; .../ckpt/<student>/<run>/checkpoint-313 -> <run>/313."""
    parts = mid.rstrip("/").split("/")
    if parts[-1].startswith("checkpoint-") and len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1].split('-', 1)[1]}"
    return parts[-1]


b = R["behavior"]
scope = R.get("args", {}).get("ablate_scope", "all")
p(f"Intervention scope: **{scope}** "
  + ("(edits only at reply positions, i.e. the answer position and generated steps; reply-position means; "
     "KL on the reply positions of held-out chats)" if scope == "reply" else "(edits at every position; all-token means)")
  + "\n")
p("## Behavior (test wordings)\n")
p("| Model | Answer-pos entropy (bits) | P(target top token) | Sampled answer entropy (mean over cats) |")
p("| --- | --- | --- | --- |")
for name, d in [("target", b["target"]), ("base", b["base"])] + list(R.get("extra_models", {}).items()):
    s = d.get("samples", {})
    sh = sum(v["H"] for v in s.values()) / max(1, len(s))
    p(f"| {short(name)} | {d['answer_pos']['H']:.2f} | {d['answer_pos']['p_top']:.2f} | {sh:.2f} |")

ex = {k: v for k, v in R.get("extra_models", {}).items() if "shift_to_target" in v}
if ex:
    sets = list(next(iter(ex.values()))["shift_to_target"])
    p("\n## Extra models: shift toward the target in the collapse neurons\n")
    p("Answer-position activations on the test wordings, projected onto the base->target difference: "
      "0 = like the base model, 1 = like the target. `all_neurons` is the whole MLP; `random@100` is a "
      "layer-matched random set. A model whose collapse neurons moved toward the target more than its other "
      "neurons did shows a higher value on the collapse sets than on `all_neurons` and `random@100`. "
      "Ablation columns: answer-position H after mean-ablating the target's top global_negH neurons in this model.\n")
    p("| Model | H | P(top) | " + " | ".join(sets) + " | H, ablate k=10 | H, ablate k=100 |")
    p("| --- " * (len(sets) + 5) + "|")
    for mid, d in ex.items():
        p(f"| {short(mid)} | {d['answer_pos']['H']:.2f} | {d['answer_pos']['p_top']:.2f} | "
          + " | ".join(f"{d['shift_to_target'][s]:+.2f}" for s in sets)
          + f" | {d['ablate_global_negH_k10']['H']:.2f} | {d['ablate_global_negH_k100']['H']:.2f} |")

p("\n## Causal: mean ablation of top-k neurons in the target (test wordings)\n")
p("| Condition | Answer-pos H | P(top) | Sampled H | Valid | KL on text |")
p("| --- | --- | --- | --- | --- | --- |")
p(f"| none | {b['target']['answer_pos']['H']:.2f} | {b['target']['answer_pos']['p_top']:.2f} | "
  f"{b['target'].get('sample_H_mean', float('nan')):.2f} | | 0 |")
for k, v in R.get("causal", {}).items():
    p(f"| {k} | {v['answer_pos']['H']:.2f} | {v['answer_pos']['p_top']:.2f} | "
      f"{v.get('sample_H_mean', float('nan')):.2f} | {v.get('valid_mean', float('nan')):.2f} | {v.get('kl_text', float('nan')):.4f} |")

if "patch" in R:
    p("\n## Activation patching between base and target (answer position)\n")
    p("| Condition | H | P(top) |\n| --- | --- | --- |")
    for k, v in R["patch"].items():
        p(f"| {k} | {v['H']:.2f} | {v['p_top']:.2f} |")

p("\n## Top neurons\n")
for d in R.get("neurons", [])[:40]:
    p(f"### {d['neuron']}  (sets: {', '.join(d['in_sets'][:6])})")
    p(f"- attribution to -H (fit): {d['attr_negH_fit']:+.3f}; to log P(top): {d['attr_lp_fit']:+.3f}")
    p(f"- answer-position activation: target {d['ans_act_target_mean']:+.3f} (percentile {100*d['ans_percentile_target']:.1f}), "
      f"base {d['ans_act_base_mean']:+.3f}; reference p1/p3/p50/p97/p99 = "
      + "/".join(f"{x:+.2f}" for x in d['ref_pct'].values()))
    p(f"- promotes: {d['promotes'][:10]}")
    p(f"- suppresses: {d['suppresses'][:10]}")
    p("- top contexts: " + " | ".join(f"`{c.strip()[:90]}` ({v:+.2f})" for v, c in d["top_ctx"][:3]))
    p("- bottom contexts: " + " | ".join(f"`{c.strip()[:90]}` ({v:+.2f})" for v, c in d["bot_ctx"][:3]))
    p("")

# figure: k vs entropy for global sets vs random
cz = R.get("causal", {})
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for setname in ["global_negH", "random_for_global_negH", "consistent_negH", "global_negH_dbase", "global_lp"]:
    pts = sorted((int(k.split("k=")[1]), v) for k, v in cz.items() if k.startswith(setname + "|"))
    if not pts:
        continue
    ax[0].plot([x for x, _ in pts], [v["answer_pos"]["H"] for _, v in pts], marker="o", label=setname)
    ax[1].plot([x for x, _ in pts], [v["kl_text"] for _, v in pts], marker="o", label=setname)
for a in ax:
    a.set_xscale("log")
    a.set_xlabel("neurons mean-ablated (k)")
ax[0].axhline(b["target"]["answer_pos"]["H"], color="k", ls=":", label="target")
ax[0].axhline(b["base"]["answer_pos"]["H"], color="gray", ls="--", label="base")
ax[0].set_ylabel("answer-position entropy (bits)")
ax[1].set_ylabel(f"KL on held-out chats, {scope} positions (nats/token)")
ax[1].set_yscale("log")
ax[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(figdir, "ablation_curve.png"), dpi=130)
p(f"\n![ablation]({os.path.basename(figdir)}/ablation_curve.png)")

open(out, "w").write("\n".join(L))
print(out)
