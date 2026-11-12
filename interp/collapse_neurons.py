"""Neuron-level search for mode-collapse mechanisms ("Neurons as an Interpretable Basis", Goodfire 2026).

A neuron is one hidden unit of a SwiGLU MLP: a = silu(W_gate x) * (W_up x), the input of down_proj.
Pipeline (one GPU, every model on the same token sequences):

  1. items     100 eval prompts (10 categories x 10 wordings) in the OLMo-2 chat format. Wordings 0-4 are the
               discovery split, 5-9 the test split. The target model (the post-trained one) answers each prompt
               greedily; the answer step (analysis/answer_tokens.answer_token) fixes the context. The answer
               position is where the model picks its answer token.
  2. reference per-neuron mean and percentiles (p1, p3, p97, p99) over held-out human chat text, plus the
               top-activating tokens of each neuron. Mean ablation uses these means.
  3. behavior  per model: answer-position entropy (bits), P(target model's top token), and sampled answer
               entropy per category (eval/extract.py on n samples per test wording at T=1).
  4. attribute target model, per item: attribution patching (activation - mean) x gradient, summed over the
               edited positions (--ablate-scope), for two metrics at the answer position: log P(top token) and negative entropy.
               Positive = the neuron pushes toward the collapsed answer / lower entropy.
               Also a base-vs-target attribution: (a_target - a_base) x gradient.
  5. causal    on the test split: mean-ablate the top-k neurons (global and per category), random
               layer-matched controls, base->target activation patching, scaling dose-response, and the
               KL on general text.

--ablate-scope sets where interventions act. reply (default): only at the positions that predict reply tokens
(the last prompt position onward, so the answer position and every generated step), with the neuron's mean over
reply positions of the reference text; attributions are summed over the same positions and the text KL is
measured on the reply positions of the reference chats. all: every position, including the system/user turn and
chat-template tokens, with the mean over all reference tokens (the behavior of job 454686). With "all", mean
ablation writes a typical-text value into template positions where many neurons sit near 0, so even one neuron
gives a large KL on text (0.32 for L12_N5747 in job 454686).
  6. describe  top neurons: percentile of their firing on the eval prompts, logit lens of their write
               direction, top-activating reference contexts, and their activation in every model.

Usage: python interp/collapse_neurons.py --target allenai/OLMo-2-0425-1B-Instruct --base allenai/OLMo-2-0425-1B \
          --models allenai/OLMo-2-0425-1B-SFT,allenai/OLMo-2-0425-1B-DPO --out $OUT_DIR/interp/olmo2_1b
"""
import argparse
import gc
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from extract import extract  # noqa: E402
from answer_tokens import answer_token  # noqa: E402

LN2 = math.log(2)
DEV = "cuda"
SCOPE = "reply"  # --ablate-scope; see the module docstring


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- model + hooks

class Neurons:
    """Hooks on every down_proj input. mode: None (pass), 'capture', 'grad'. Interventions edit activations.

    Edits apply from sequence position `start` on (set per forward by the callers, see edit_start). A forward of
    one token is a cached decoding step, past any prompt, so it is always edited."""

    def __init__(self, model):
        self.model = model
        model.neurons = self  # lets answer_pos / sample_answers / text_kl set the edit start
        self.layers = model.model.layers
        self.L = len(self.layers)
        self.N = self.layers[0].mlp.down_proj.in_features
        self.mode = None
        self.acts = [None] * self.L
        self.edits = {}  # layer -> list of (idx tensor, fn(a_sub) -> new a_sub); a_sub is [B, T - start, k]
        self.start = 0
        self.handles = [l.mlp.down_proj.register_forward_pre_hook(self._hook(i)) for i, l in enumerate(self.layers)]

    def _hook(self, i):
        def fn(mod, args):
            a = args[0]
            if i in self.edits:
                a = a.clone()
                s = 0 if a.shape[1] == 1 else self.start
                for idx, f in self.edits[i]:
                    a[:, s:, idx] = f(a[:, s:, idx])
            if self.mode == "grad":
                a.retain_grad()  # the graph is connected: attribute() feeds inputs_embeds that require grad
                self.acts[i] = a
            elif self.mode == "capture":
                self.acts[i] = a.detach()
            return (a,)
        return fn

    def clear(self):
        self.edits, self.mode = {}, None
        self.acts = [None] * self.L

    def set_mean_ablation(self, neurons, mean):
        """neurons: list of (layer, idx). mean: [L, N] tensor."""
        self.edits = {}
        by = defaultdict(list)
        for l, n in neurons:
            by[l].append(n)
        for l, ns in by.items():
            idx = torch.tensor(ns, device=DEV)
            m = mean[l, ns].to(DEV)  # list index: works whether mean is on cpu or gpu
            self.edits[l] = [(idx, lambda a, m=m: m.to(a.dtype).expand_as(a))]

    def set_scale(self, neurons, mean, alpha):
        """a -> mean + alpha * (a - mean): alpha 0 is mean ablation, 1 is unchanged, >1 amplifies."""
        self.edits = {}
        by = defaultdict(list)
        for l, n in neurons:
            by[l].append(n)
        for l, ns in by.items():
            idx = torch.tensor(ns, device=DEV)
            m = mean[l, ns].to(DEV)  # list index: works whether mean is on cpu or gpu
            self.edits[l] = [(idx, lambda a, m=m: m.to(a.dtype) + alpha * (a - m.to(a.dtype)))]

    def set_values(self, neurons, values):
        """Replace with given per-position values. values: {layer: [T, len(ns)] tensor}; sequence length must match.
        Only the edited positions (start on) are replaced: the last T - start rows of the values."""
        self.edits = {}
        by = defaultdict(list)
        for l, n in neurons:
            by[l].append(n)
        for l, ns in by.items():
            idx = torch.tensor(ns, device=DEV)
            v = values[l]
            self.edits[l] = [(idx, lambda a, v=v: v[v.shape[0] - a.shape[1]:].to(a.dtype).expand_as(a))]


def edit_start(model, prompt_len):
    """Set where the model's interventions start for a sequence whose reply begins after prompt_len tokens:
    the position that predicts the first reply token (scope reply), or 0 (scope all). Returns that position."""
    s = max(0, prompt_len - 1) if SCOPE == "reply" else 0
    hk = getattr(model, "neurons", None)
    if hk is not None:
        hk.start = s
    return s


def load(mid, dtype=torch.bfloat16):
    log("loading", mid)
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=dtype).to(DEV).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def entropy_bits(logits):
    lp = F.log_softmax(logits.float(), -1)
    return -(lp.exp() * lp).sum(-1) / LN2


# ---------------------------------------------------------------- items

def build_items(tok, target, H, cats, args):
    items = []
    for cname, cat in cats.items():
        for pi, p in enumerate(cat["prompts"]):
            text = f"{p} {cat['instruction']}"
            chat = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                           add_generation_prompt=True)
            ids = tok(chat, add_special_tokens=False)["input_ids"]
            items.append(dict(cat=cname, pidx=pi, split="fit" if pi < 5 else "test", prompt=text, ids=ids))
    # greedy reply of the target model -> answer step
    for it in items:
        x = torch.tensor([it["ids"]], device=DEV)
        with torch.no_grad():
            out = target.generate(x, max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)
        rep = out[0, len(it["ids"]):].tolist()
        raw = tok.decode(rep, skip_special_tokens=True)
        ans, ok = extract(raw, cats[it["cat"]], it["cat"])
        j = answer_token(rep, tok, cats[it["cat"]], it["cat"], ans) if ok else None
        j = 0 if j is None else j
        it.update(reply=raw, answer=ans, valid=ok, ans_step=j, ctx=it["ids"] + rep[:j], ans_tok=rep[j] if j < len(rep) else None)
    return items


# ---------------------------------------------------------------- reference stats

def reference_stats(model, H, tok, path, n_docs, max_len, n_top=8):
    docs = [json.loads(l) for l in open(path)][:n_docs]
    seqs, plens = [], []
    for d in docs:
        user = [{"role": "user", "content": d["prompt"]}]
        msgs = user + [{"role": "assistant", "content": d.get("response") or d.get("reply") or ""}]
        s = tok(tok.apply_chat_template(msgs, tokenize=False), add_special_tokens=False)["input_ids"]
        pr = tok(tok.apply_chat_template(user, tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]
        if s[:len(pr)] != pr:
            raise SystemExit("chat template: the prompt with generation prompt is not a prefix of the full chat")
        seqs.append(s[:max_len])
        plens.append(len(pr))
    # reply positions: the ones that predict reply tokens (prompt_len - 1 on); a doc whose prompt fills max_len has none
    starts = [max(0, pl - 1) for pl in plens]
    acts = [[] for _ in range(H.L)]
    H.mode = "capture"
    for s in seqs:
        with torch.no_grad():
            model(torch.tensor([s], device=DEV))
        for l in range(H.L):
            acts[l].append(H.acts[l][0].half().cpu())
    H.clear()
    flat_tok = [t for s in seqs for t in s]
    doc_of = [(di, ti) for di, s in enumerate(seqs) for ti in range(len(s))]
    qs = torch.tensor([0.01, 0.03, 0.5, 0.97, 0.99])
    mean = torch.zeros(H.L, H.N)
    mean_reply = torch.zeros(H.L, H.N)
    n_reply = sum(max(0, len(s) - st) for s, st in zip(seqs, starts))
    if n_reply == 0:
        raise SystemExit("no reply positions in the reference text (prompts longer than --max-len?)")
    pct = torch.zeros(H.L, len(qs), H.N)
    top_v = torch.zeros(H.L, H.N, n_top)
    top_i = torch.zeros(H.L, H.N, n_top, dtype=torch.long)
    bot_v = torch.zeros(H.L, H.N, n_top)
    bot_i = torch.zeros(H.L, H.N, n_top, dtype=torch.long)
    sorted_acts = []
    for l in range(H.L):
        A = torch.cat(acts[l]).to(DEV).float()  # [T, N]
        mean[l] = A.mean(0).cpu()
        mean_reply[l] = (sum(x[st:].to(DEV).float().sum(0) for x, st in zip(acts[l], starts)) / n_reply).cpu()
        S, _ = A.sort(0)
        T = S.shape[0]
        pct[l] = S[(qs * (T - 1)).round().long()].cpu()
        v, i = A.topk(n_top, 0)
        top_v[l], top_i[l] = v.T.cpu(), i.T.cpu()
        v, i = (-A).topk(n_top, 0)
        bot_v[l], bot_i[l] = (-v).T.cpu(), i.T.cpu()
        sorted_acts.append(S.half().cpu())
        acts[l] = None
    return dict(mean=mean, mean_reply=mean_reply, n_reply=n_reply, pct=pct, top_v=top_v, top_i=top_i, bot_v=bot_v,
                bot_i=bot_i, tokens=flat_tok, doc_of=doc_of, seqs=seqs, plens=plens, sorted=sorted_acts)


def percentile_of(sorted_col, v):
    return float(torch.searchsorted(sorted_col.float().contiguous(), torch.tensor([float(v)])).item()) / len(sorted_col)


# ---------------------------------------------------------------- behavior

@torch.no_grad()
def answer_pos(model, items, which=None):
    """Answer-position logits stats. Returns per item: entropy bits, logp of target top token, top5 tokens."""
    out = []
    for it in items:
        edit_start(model, len(it["ids"]))
        lg = model(torch.tensor([it["ctx"]], device=DEV)).logits[0, -1].float()
        lp = F.log_softmax(lg, -1)
        out.append(dict(H=float(entropy_bits(lg)), lp=float(lp[it["top"]]),
                        top=lp.topk(5).indices.tolist(), topp=lp.topk(5).values.exp().tolist()))
    return out


@torch.no_grad()
def sample_answers(model, tok, items, cats, n, max_new=24):
    res = defaultdict(list)
    for it in items:
        x = torch.tensor([it["ids"]], device=DEV).repeat(n, 1)
        edit_start(model, len(it["ids"]))
        o = model.generate(x, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=max_new,
                           pad_token_id=tok.eos_token_id)
        for r in tok.batch_decode(o[:, x.shape[1]:], skip_special_tokens=True):
            a, ok = extract(r, cats[it["cat"]], it["cat"])
            res[it["cat"]].append(a if ok else None)
    stats = {}
    for c, ans in res.items():
        v = [a for a in ans if a is not None]
        cnt = Counter(v)
        Hc = -sum(k / len(v) * math.log2(k / len(v)) for k in cnt.values()) if v else float("nan")
        stats[c] = dict(H=Hc, valid=len(v) / len(ans), top=cnt.most_common(5), n=len(ans))
    return stats


def summarize_ap(ap, items, split="test"):
    sel = [i for i, it in enumerate(items) if it["split"] == split]
    return dict(H=float(np.mean([ap[i]["H"] for i in sel])), p_top=float(np.mean([math.exp(ap[i]["lp"]) for i in sel])))


@torch.no_grad()
def text_kl(model, H, seqs, plens, ref_logits=None):
    """Mean per-token KL(ref || model) on held-out chats, over the edited positions (scope reply: the reply
    positions; all: every position). Returns (kl, logits list)."""
    kls, outs = [], []
    for k, s in enumerate(seqs):
        st = edit_start(model, plens[k])
        lg = model(torch.tensor([s], device=DEV)).logits[0].float()
        if ref_logits is None:
            outs.append(F.log_softmax(lg, -1).half().cpu())
        elif st < len(s):
            r = ref_logits[k].to(DEV).float()[st:]
            kls.append(float((r.exp() * (r - F.log_softmax(lg[st:], -1))).sum(-1).mean()))
    return (float(np.mean(kls)) if kls else 0.0), outs


# ---------------------------------------------------------------- attribution

def attribute(model, H, items, mean, base_acts=None):
    """Per item [L, N] attributions (summed over the positions the interventions edit, see edit_start) for metric
    lp (log P top) and negH (-entropy bits), plus answer-position-only versions, plus base->target versions if
    base_acts is given."""
    L, N = H.L, H.N
    keys = ["lp", "negH", "lp_ans", "negH_ans"] + (["lp_dbase", "negH_dbase"] if base_acts is not None else [])
    out = {k: np.zeros((len(items), L, N), dtype=np.float16) for k in keys}
    mean_d = mean.to(DEV)
    for ii, it in enumerate(items):
        st = edit_start(model, len(it["ids"]))
        for metric in ["lp", "negH"]:
            H.mode = "grad"
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                emb = model.get_input_embeddings()(torch.tensor([it["ctx"]], device=DEV)).detach().requires_grad_(True)
                lg = model(inputs_embeds=emb).logits[0, -1].float()
                lp = F.log_softmax(lg, -1)
                m = lp[it["top"]] if metric == "lp" else (lp.exp() * lp).sum() / LN2
                m.backward()
            for l in range(L):
                a = H.acts[l]  # [1, T, N]; index after .grad (a view of it has no .grad)
                g = a.grad[0].float()
                af = a.detach()[0].float()
                d = (af - mean_d[l].float()) * g  # [T, N]
                out[metric][ii, l] = d[st:].sum(0).cpu().numpy()
                out[metric + "_ans"][ii, l] = d[-1].cpu().numpy()
                if base_acts is not None:
                    db = (af - base_acts[ii][l].to(DEV).float()) * g
                    out[metric + "_dbase"][ii, l] = db[st:].sum(0).cpu().numpy()
            H.clear()
    return out


@torch.no_grad()
def capture(model, H, items):
    acts = []
    H.mode = "capture"
    for it in items:
        model(torch.tensor([it["ctx"]], device=DEV))
        acts.append([H.acts[l][0].half().cpu() for l in range(H.L)])
    H.clear()
    return acts


# ---------------------------------------------------------------- main

def rank(score, k):
    """score [L, N] -> list of (l, n) with the largest values."""
    flat = torch.from_numpy(np.asarray(score, dtype=np.float32)).flatten()
    idx = flat.topk(k).indices
    N = score.shape[1]
    return [(int(i) // N, int(i) % N) for i in idx]


def random_matched(neurons, N, rng, exclude):
    out = []
    for l, _ in neurons:
        while True:
            n = rng.randrange(N)
            if (l, n) not in exclude:
                out.append((l, n))
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--models", default="", help="extra models (comma list) for behavior + neuron activations")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=os.path.expandvars("$OUT_DIR/data/human/heldout.jsonl"))
    ap.add_argument("--ref-docs", type=int, default=160)
    ap.add_argument("--kl-docs", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--samples", type=int, default=50, help="samples per test wording")
    ap.add_argument("--ks", default="1,3,10,30,100,300")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablate-scope", default="reply", choices=["reply", "all"],
                    help="positions the interventions edit (and attributions sum over); see the module docstring")
    args = ap.parse_args()
    global SCOPE
    SCOPE = args.ablate_scope
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    cats = yaml.safe_load(open(os.path.join(ROOT, "eval", "categories.yaml")))
    tok = AutoTokenizer.from_pretrained(args.target)
    ks = [int(k) for k in args.ks.split(",")]
    R = {"args": vars(args)}

    def dump():
        json.dump(R, open(os.path.join(args.out, "results.json"), "w"), indent=1, default=str)

    target = load(args.target)
    H = Neurons(target)
    log(f"{H.L} layers x {H.N} neurons")

    # 1. items
    items = build_items(tok, target, H, cats, args)
    with torch.no_grad():
        for it in items:
            lg = target(torch.tensor([it["ctx"]], device=DEV)).logits[0, -1]
            it["top"] = int(lg.argmax())
            it["top_str"] = tok.decode([it["top"]])
    R["items"] = [{k: v for k, v in it.items() if k not in ("ids", "ctx")} | {"ctx_len": len(it["ctx"])} for it in items]
    log("items:", Counter((it["cat"], it["top_str"]) for it in items).most_common(30))
    dump()

    # 2. reference stats (target), KL baseline text
    ref = reference_stats(target, H, tok, args.ref, args.ref_docs + args.kl_docs, args.max_len)
    kl_seqs = ref["seqs"][args.ref_docs:]
    kl_plens = ref["plens"][args.ref_docs:]
    mean = ref["mean_reply"] if SCOPE == "reply" else ref["mean"]
    _, kl_ref = text_kl(target, H, kl_seqs, kl_plens)
    R["ref"] = dict(scope=SCOPE, reply_positions=ref["n_reply"], positions=len(ref["tokens"]),
                    kl_docs_with_reply=sum(len(s) > max(0, pl - 1) for s, pl in zip(kl_seqs, kl_plens)))
    log("reference stats done", R["ref"])

    # 3. behavior of the target, unmodified
    ap0 = answer_pos(target, items)
    for it, a in zip(items, ap0):
        it["ap"] = a
    R["behavior"] = {"target": dict(answer_pos=summarize_ap(ap0, items), answer_pos_fit=summarize_ap(ap0, items, "fit"),
                                    samples=sample_answers(target, tok, [i for i in items if i["split"] == "test"], cats, args.samples))}
    log("target behavior", R["behavior"]["target"]["answer_pos"])
    dump()

    # base activations (same contexts) for the base->target attribution and patching
    base = load(args.base)
    HB = Neurons(base)
    base_acts = capture(base, HB, items)
    apb = answer_pos(base, items)
    R["behavior"]["base"] = dict(answer_pos=summarize_ap(apb, items),
                                 samples=sample_answers(base, tok, [i for i in items if i["split"] == "test"], cats, args.samples))
    log("base behavior", R["behavior"]["base"]["answer_pos"])
    dump()

    # 4. attribution on the target
    A = attribute(target, H, items, mean, base_acts)
    np.savez_compressed(os.path.join(args.out, "attributions.npz"), **A,
                        cats=np.array([it["cat"] for it in items]), split=np.array([it["split"] for it in items]))
    fit = np.array([it["split"] == "fit" for it in items])
    catv = np.array([it["cat"] for it in items])
    log("attribution done")

    # neuron sets from the discovery split
    sets = {}
    for key in ["negH", "lp", "negH_dbase", "lp_dbase", "negH_ans"]:
        sets[f"global_{key}"] = rank(A[key][fit].astype(np.float32).mean(0), max(ks))
    # consistent: mean / std across categories of category means (neurons that push collapse on every question)
    cm = np.stack([A["negH"][fit & (catv == c)].astype(np.float32).mean(0) for c in cats])  # [C, L, N]
    sets["consistent_negH"] = rank(cm.mean(0) / (cm.std(0) + 1e-3 * np.abs(cm.mean(0)).max()) * (cm.min(0) > 0), max(ks))
    for c in cats:
        sets[f"cat_{c}"] = rank(A["lp"][fit & (catv == c)].astype(np.float32).mean(0), 50)
    R["sets"] = {k: v[:50] for k, v in sets.items()}
    dump()

    # 5. causal tests on the test split
    test_items = [i for i in items if i["split"] == "test"]
    causal = {}

    def eval_cond(name, setup, do_samples=True, kl=True):
        setup()
        r = dict(answer_pos=summarize_ap(answer_pos(target, items), items))
        if kl:
            r["kl_text"] = text_kl(target, H, kl_seqs, kl_plens, kl_ref)[0]
        if do_samples:
            s = sample_answers(target, tok, test_items, cats, args.samples)
            r["samples"] = s
            r["sample_H_mean"] = float(np.mean([v["H"] for v in s.values()]))
            r["valid_mean"] = float(np.mean([v["valid"] for v in s.values()]))
        H.clear()
        causal[name] = r
        log(name, r["answer_pos"], r.get("sample_H_mean"), r.get("kl_text"))
        R["causal"] = causal
        dump()

    s0 = R["behavior"]["target"]["samples"]
    R["behavior"]["target"]["sample_H_mean"] = float(np.mean([v["H"] for v in s0.values()]))
    for setname in ["global_negH", "consistent_negH", "global_negH_dbase", "global_lp"]:
        for k in ks:
            nr = sets[setname][:k]
            eval_cond(f"{setname}|k={k}", lambda nr=nr: H.set_mean_ablation(nr, mean), do_samples=k in (10, 100, 300))
            rnd = random_matched(nr, H.N, rng, set(nr))
            if setname == "global_negH":
                eval_cond(f"random_for_{setname}|k={k}", lambda rnd=rnd: H.set_mean_ablation(rnd, mean),
                          do_samples=k in (10, 100, 300))
    # dose-response: scale the top neurons
    for k in (1, 10):
        for alpha in (-1.0, 0.0, 0.5, 2.0, 4.0):
            nr = sets["global_negH"][:k]
            eval_cond(f"scale_global_negH|k={k}|alpha={alpha}", lambda nr=nr, alpha=alpha: H.set_scale(nr, mean, alpha),
                      do_samples=(k == 10 and alpha in (-1.0, 4.0)), kl=True)
    # per-category specificity: ablate category c's top-k favorite neurons, measure every category
    spec = {}
    for c in cats:
        for k in (1, 5, 20):
            H.set_mean_ablation(sets[f"cat_{c}"][:k], mean)
            a = answer_pos(target, items)
            H.clear()
            spec[f"{c}|k={k}"] = {c2: dict(
                d_lp=float(np.mean([a[i]["lp"] - ap0[i]["lp"] for i, it in enumerate(items) if it["split"] == "test" and it["cat"] == c2])),
                d_H=float(np.mean([a[i]["H"] - ap0[i]["H"] for i, it in enumerate(items) if it["split"] == "test" and it["cat"] == c2])))
                for c2 in cats}
    R["specificity"] = spec
    dump()
    # base -> target patching of top-k neurons (answer position metrics only; same contexts)
    patch = {}
    for setname in ["global_negH", "global_negH_dbase"]:
        for k in ks:
            nr = sets[setname][:k]
            res = []
            for ii, it in enumerate(items):
                vals = {}
                for l, n in nr:
                    vals.setdefault(l, []).append(n)
                v = {l: base_acts[ii][l][:, ns].to(DEV) for l, ns in vals.items()}
                H.set_values([(l, n) for l, ns in vals.items() for n in ns], v)
                res.append(answer_pos(target, [it])[0])
                H.clear()
            patch[f"{setname}|k={k}"] = summarize_ap(res, items)
            log("patch base->target", setname, k, patch[f"{setname}|k={k}"])
    # reverse: target -> base
    tgt_acts = capture(target, H, items)
    for setname in ["global_negH", "global_negH_dbase"]:
        for k in ks:
            nr = sets[setname][:k]
            res = []
            for ii, it in enumerate(items):
                vals = {}
                for l, n in nr:
                    vals.setdefault(l, []).append(n)
                v = {l: tgt_acts[ii][l][:, ns].to(DEV) for l, ns in vals.items()}
                HB.set_values([(l, n) for l, ns in vals.items() for n in ns], v)
                res.append(answer_pos(base, [it])[0])
                HB.clear()
            patch[f"target->base|{setname}|k={k}"] = summarize_ap(res, items)
            log("patch target->base", setname, k, patch[f"target->base|{setname}|k={k}"])
    R["patch"] = patch
    dump()

    # 6. describe the top neurons
    W_U = target.lm_head.weight.float()
    fn = target.model.norm.weight.float()
    desc_set = []
    for sname, kk in [("global_negH", 20), ("global_negH_dbase", 10), ("consistent_negH", 10)] + [(f"cat_{c}", 3) for c in cats]:
        for nrn in sets[sname][:kk]:
            if nrn not in desc_set:
                desc_set.append(nrn)
    tokens = ref["tokens"]

    def ctx_str(flat_i, w=12):
        di, ti = ref["doc_of"][flat_i]
        s = ref["seqs"][di]
        return tok.decode(s[max(0, ti - w):ti]) + "[[" + tok.decode([s[ti]]) + "]]" + tok.decode(s[ti + 1:ti + 4])

    D = []
    for l, n in desc_set:
        mlp = target.model.layers[l]
        w = mlp.mlp.down_proj.weight[:, n].float()
        pf = getattr(mlp, "post_feedforward_layernorm", None)
        if pf is not None:
            w = w * pf.weight.float()
        ll = W_U @ (w * fn)
        av = np.array([float(tgt_acts[ii][l][-1, n]) for ii in range(len(items))])
        bv = np.array([float(base_acts[ii][l][-1, n]) for ii in range(len(items))])
        col = ref["sorted"][l][:, n]
        D.append(dict(
            neuron=f"L{l}_N{n}", layer=l, idx=n,
            in_sets=[s for s in sets if (l, n) in sets[s][:50]],
            attr_negH_fit=float(A["negH"][fit][:, l, n].astype(np.float32).mean()),
            attr_lp_fit=float(A["lp"][fit][:, l, n].astype(np.float32).mean()),
            attr_negH_by_cat={c: float(A["negH"][catv == c][:, l, n].astype(np.float32).mean()) for c in cats},
            ref_mean=float(mean[l, n]), ref_pct={q: float(ref["pct"][l, i, n]) for i, q in enumerate(["p1", "p3", "p50", "p97", "p99"])},
            ans_act_target_mean=float(av.mean()), ans_act_base_mean=float(bv.mean()),
            ans_act_target_by_cat={c: float(av[catv == c].mean()) for c in cats},
            ans_act_base_by_cat={c: float(bv[catv == c].mean()) for c in cats},
            ans_percentile_target=float(np.mean([percentile_of(col, v) for v in av])),
            promotes=[tok.decode([t]) for t in ll.topk(15).indices.tolist()],
            suppresses=[tok.decode([t]) for t in (-ll).topk(15).indices.tolist()],
            top_ctx=[(float(v), ctx_str(int(i))) for v, i in zip(ref["top_v"][l, n][:5], ref["top_i"][l, n][:5])],
            bot_ctx=[(float(v), ctx_str(int(i))) for v, i in zip(ref["bot_v"][l, n][:5], ref["bot_i"][l, n][:5])],
        ))
    R["neurons"] = D
    dump()

    # activations of the described neurons + behavior in the extra models (training stages, students)
    extra = {}
    del base, HB
    gc.collect()  # model.neurons <-> Neurons.model is a reference cycle
    torch.cuda.empty_cache()
    for mid in [m for m in args.models.split(",") if m]:
        m = load(mid)
        Hm = Neurons(m)
        ma = capture(m, Hm, items)
        apm = answer_pos(m, items)
        extra[mid] = dict(answer_pos=summarize_ap(apm, items),
                          samples=sample_answers(m, tok, test_items, cats, args.samples),
                          neuron_ans_act={d["neuron"]: float(np.mean([float(ma[ii][d["layer"]][-1, d["idx"]]) for ii in range(len(items))]))
                                          for d in D})
        # does mean-ablating the target's global_negH top-k neurons also de-collapse this model?
        for k in (10, 100):
            Hm.set_mean_ablation(sets["global_negH"][:k], mean)
            extra[mid][f"ablate_global_negH_k{k}"] = summarize_ap(answer_pos(m, items), items)
            Hm.clear()
        log("extra", mid, extra[mid]["answer_pos"])
        R["extra_models"] = extra
        dump()
        del m, Hm
        gc.collect()
        torch.cuda.empty_cache()
    log("done")


if __name__ == "__main__":
    main()
