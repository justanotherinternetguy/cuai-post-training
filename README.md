# unicorn

Teacher → student SFT experiments on Cornell's Unicorn cluster. Does a base model fine-tuned on a post-trained teacher's outputs inherit the teacher's preferences (mode collapse)?

## Layout

```
env.sh              paths + env vars; every job sources this
requirements.txt    one env for generation, SFT and interp (vLLM pins torch)
configs/models.yaml model registry (teachers, students)
scripts/            download_models.py, check_model.py, smoke_test.py, infer_test.py
slurm/              job scripts (setup_env, download, smoke_test, check_model, infer_test, eval_prefs)
eval/               preference eval: categories.yaml, normalize.py, prefs_eval.py
gen/ train/ analysis/   research code (not written yet)
sync.sh             push local code to the cluster
```

## Where things live on the cluster

| What | Path | Why |
| --- | --- | --- |
| Code, Slurm logs | `~/unicorn` | small |
| Conda env | `~/envs/unicorn` | ~15 GB |
| HF cache (weights) | `/share/cuvl/qx235/hf` | ~300 GB; home NFS is 94% full |
| Outputs (data, ckpts, wandb) | `/share/cuvl/qx235/unicorn` | large |
| HF token | `~/.cache/huggingface/token` | private; `/share/cuvl` is group-readable |

## Common commands (on the login node, from `~/unicorn`)

```bash
source env.sh
sbatch slurm/setup_env.sub                                 # build/refresh the env (CPU job)
sbatch --requeue --array=0-4 slurm/download.sub teachers   # download all teachers
sbatch slurm/download.sub teachers qwen3_32b               # one model
slurm/submit_gpu.sh slurm/smoke_test.sub                   # vLLM + TRL + nnsight + TransformerLens on a 135M model
slurm/submit_gpu.sh --requeue slurm/check_model.sub Qwen/Qwen3-32B   # load a teacher on 2x A6000, sample 20 answers
sbatch slurm/download.sub other qwen38_27b                 # Qwen3.8-27B (outside the teacher set)
slurm/submit_gpu.sh --requeue slurm/infer_test.sub Qwen/Qwen3.8-27B   # basic chat inference test, thinking off + on
slurm/submit_gpu.sh --requeue slurm/eval_prefs.sub --model Qwen/Qwen3-32B --name qwen3_32b   # preference eval
squeue --me ; tail -f logs/<job>.out
```

Interactive GPU shell: `salloc -p gpu --gres=gpu:nvidia_rtx_a6000:2 --cpus-per-task=4 --mem=48g -t 2:00:00`

## Preference eval

`eval/prefs_eval.py` measures how concentrated a model's answers are on "pick one" questions.
10 categories x 10 paraphrases (`eval/categories.yaml`): number 1-10, number 1-100, vacation country,
city, color, animal, first name, letter A-Z, fruit, coin flip.

- **sample** mode: 100 samples per prompt at T=1, no top-p/top-k (`generation_config="vllm"` ignores the
  model's defaults). Answers are normalized (`eval/normalize.py`): first line, lowercase, no punctuation,
  markdown or articles, cut at the first comma, synonyms mapped (USA -> united states), number words parsed.
- **logprob** mode (all categories): the exact probability of every answer the model would give, with no
  candidate list. It walks the tree of the model's own continuations: each step reads the top-k (`--topk`, 100)
  next tokens of every open branch and splits the branch's probability over them. A branch ends at end-of-turn,
  when its first line is complete, or at `--max-tokens`. Finished branches are normalized like samples, so
  "Blue", "**Blue**" and "My favorite color is blue!" all count as blue. This is the distribution sample mode
  draws from, without sampling noise. Branches below `--min-prob` (1e-4) are dropped; `pruned` is the mass lost.
- **Prompt format**: `chat` (tokenizer's chat template, thinking disabled, a reply ends at end-of-turn) or
  `base` (`Question: ...\nAnswer:`, a reply ends at the first newline). `auto` picks chat when the model has a template.
  Use `--format base` on base models and SFT students so they all see the same prompt.
- **Answer sets**: only questions that name their own options are closed (numbers, letters, heads/tails).
  City, country, color, animal, name and fruit are open: any answer of at most 4 words counts, and only
  spellings and plurals are merged (`synonyms`). Color had a 27-color list until 2026-11-05.
- Outputs in `$OUT_DIR/eval/prefs/<name>/`: `samples.jsonl` (raw + normalized), `logprobs.jsonl` (every
  finished branch: raw text, answer, probability, node id), `tree.jsonl.gz` (every step of the logprob tree:
  the full top-k next-token distribution at that point, the answer each token leads to, and the answers
  reachable from there; format in `write_tree` in `eval/prefs_eval.py`), `summary.json` (pooled entropy in bits, top-1 share, top
  answers, per-paraphrase top-1, mean per-paraphrase entropy, valid rate, pruned mass). The job also prints a table.
- **Dashboard**: `python analysis/export_dashboard.py <copy of $OUT_DIR/eval/prefs> <out dir>` turns each
  model's outputs into the JSON files of the "Teacher answer distributions" dashboard. Its token view keeps every
  top-100 token at the first step and wherever the answer is chosen, and trims the rest; `tree.jsonl.gz` has all.
- `--dry-run` loads only the tokenizer and prints rendered prompts and stop tokens. It is light enough
  for the login node and is the quickest way to check a new model's chat template.

## Cluster notes

- **Submit GPU jobs with `slurm/submit_gpu.sh`, not plain `sbatch`.** About half the GPU nodes have an NVIDIA driver that only supports CUDA 12.8, and our env (vLLM 0.33, torch cu130) needs driver >= 580. The wrapper excludes those nodes. For `salloc`, pass the same `--exclude` list.
- 32B teachers in bf16 (~65 GB) need 2x 48 GB GPUs (A6000 / 6000 Ada / L40S) with TP=2, or one 80 GB+ GPU (A100 80GB, H100 NVL, RTX Pro 6000).
- Ask for modest CPU/RAM (4 CPUs, 48 GB) on GPU jobs. Larger requests sit in the queue because nodes with free GPUs are often short on CPUs or RAM, and Slurm still reports `Priority` as the reason.
- vLLM scripts need an `if __name__ == "__main__":` guard and `VLLM_WORKER_MULTIPROC_METHOD=spawn` (set in `env.sh`).
- `sbatch` sometimes times out but still submits. Run `squeue --me` before you resubmit.
- `cuvl` partition (high priority) is open to us, but it's mostly 2080 Ti / 1080 Ti GPUs plus one 4x A6000 node (`badfellow`).
- wandb: run `wandb login` once on the login node before logging runs.
