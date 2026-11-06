# Teacher preference eval: results

Run on 2026-11-06 on the Cornell Unicorn cluster (user `qx235`). Report written 2026-11-06 by `analysis/make_report.py` from the outputs in `/share/cuvl/qx235/unicorn/eval/prefs/`. Setup tests and history: `teacher_tests_report.md`.

Every teacher answered every wording of every question in two ways:

- **Sampling:** 100 full replies per wording at temperature 1. The answer distribution is the share of replies giving each answer.
- **Token-tree walk:** the exact probability of every answer, computed from the model's next-token probabilities without sampling (see 1.4).

## 1. Experiment

### 1.1 Research question

Does a base model fine-tuned (SFT) on a post-trained teacher's outputs inherit the teacher's preferences, in particular its mode collapse on open "pick one" questions? This stage measures the **teachers** only. The same eval will run on the student base models (before SFT) and on the SFT students, with the same prompts.

### 1.2 Models and jobs

| Teacher | HF id | Snapshot | Job | Node | Load time | Job time |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-32B | `Qwen/Qwen2.5-32B-Instruct` | `5ede1c97` | 14789 | barcheck-compute-01 | 1010 s | 00:18:58 |
| Qwen3-32B | `Qwen/Qwen3-32B` | `9216db57` | 14986 | barcheck-compute-01 | 364 s | 00:08:12 |
| OLMo-2-32B | `allenai/OLMo-2-0325-32B-Instruct` | `b9602434` | 14987 | lil-compute-04 | 1545 s | 00:29:50 |
| Gemma-3-27B | `google/gemma-3-27b-it` | `005ad340` | 14793 | lil-compute-04 | 1111 s | 00:20:15 |
| Mistral-3.1-24B | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | `68faf511` | 14794 | barcheck-compute-01 | 1347 s | 00:26:54 |

### 1.3 Settings

| Item | Value |
| --- | --- |
| GPUs per job | 2x NVIDIA RTX A6000 (48 GB), tensor parallel = 2 |
| Partition | `gpu`, 4 CPUs, 48 GB RAM, 4 h limit, `--requeue` |
| Engine | vLLM 0.33.0, `max_model_len=2048`, `gpu_memory_utilization=0.90`, `generation_config="vllm"` |
| Prompt format | `chat`: the model's chat template, thinking off, a reply ends at end-of-turn |
| Sampling | 100 replies per wording, T=1.0, top-p 1.0, no top-k, max 32 tokens, seed 0 |
| Token-tree walk | top-100 next tokens per step, branches below P=0.0001 dropped, max depth 32 tokens |
| Code | `eval/prefs_eval.py`, `eval/normalize.py`, `eval/categories.yaml`, `slurm/eval_prefs.sub` |
| Command | `sbatch --exclude=<old-driver nodes> --requeue -t 4:00:00 slurm/eval_prefs.sub --model <hf id> --name <name> --out-root <dir>` (both modes, the default) |
| Outputs | `/share/cuvl/qx235/unicorn/eval/prefs/<name>/`: `samples.jsonl`, `samples_tree.jsonl.gz`, `logprobs.jsonl`, `tree.jsonl.gz`, `summary.json`; `run_info.json` (jobs) |

End-of-turn token per model: Qwen2.5-32B `<|im_end|>`, Qwen3-32B `<|im_end|>`, OLMo-2-32B `<|endoftext|>`, Gemma-3-27B `<end_of_turn>`, Mistral-3.1-24B `</s>`.

### 1.4 Method

**Sampling.** Each wording gets 100 replies at T=1, so 1,000 per question and 10,000 per model. Each reply is normalized to an answer (below) and the shares are counted. At every step of every reply, the model's top-100 next tokens are also logged (`samples_tree.jsonl.gz`).

**Token-tree walk.** For each wording:

1. Start at the prompt with probability 1. Ask the model for its top-100 next tokens and their probabilities.
2. Each token opens a branch. A branch's probability is the product of the token probabilities along it.
3. Ask again for every open branch (all open branches go to vLLM as one batch per step). A branch ends at end-of-turn, when its first line is complete, or after 32 tokens.
4. Branches below P=0.0001 are dropped. The dropped share is reported as *pruned*.
5. Each finished branch is a full reply. It is normalized like a sample, and replies with the same answer are added up ("Blue" and "Blue." both count as blue).

This gives the distribution that sampling draws from, without sampling noise. Every model call returns the same probabilities for the same input, so each branch is asked once.

**Pooling.** A question's distribution is the mean over its 10 wordings (each wording weighted equally, like pooling 10 equal-sized sample sets). Per-wording numbers are reported too.

**Normalization** (`eval/normalize.py`): first line only, lowercase, strip markdown, quotes and edge punctuation, drop prefixes like "my answer is", cut at the first comma, drop leading articles, apply the category's synonym map. Numbers: digits or number words. Letters: a single letter, optionally "the letter X".

**Valid:** closed questions (numbers, letter, coin flip), the answer is in the answer set. Open questions, a non-empty answer of at most 4 words. Invalid answers (refusals, empty replies) are reported separately and left out of the distribution.

### 1.5 Metrics

| Metric | Definition |
| --- | --- |
| Entropy (H), bits | Shannon entropy of the pooled answer distribution. 0 = always the same answer. Max for closed sets = log2(size of the set). |
| Top-1 share | Share of valid answers that are the most common answer |
| Per-wording H | Entropy for each wording on its own, averaged over the 10 wordings. Leaves out the variety that comes only from different wordings. |
| Valid | Share of replies (or probability) that are valid |
| Distinct | Number of different valid answers with any probability |
| Pruned | Tree walk only: probability dropped because a branch fell below the cutoff, mean (max) over the 10 wordings |

### 1.6 Questions

| Question | Kind | Answer set | Instruction |
| --- | --- | --- | --- |
| `number_1_10` | int | integers 1-10 | Answer with just the number. |
| `number_1_100` | int | integers 1-100 | Answer with just the number. |
| `vacation_country` | open | open (any answer of at most 4 words) | Answer with just the name of the country. |
| `city` | open | open (any answer of at most 4 words) | Answer with just the name of the city. |
| `color` | open | open (any answer of at most 4 words) | Answer with one word. |
| `animal` | open | open (any answer of at most 4 words) | Answer with one word. |
| `first_name` | open | open (any answer of at most 4 words) | Answer with just the first name. |
| `letter` | letter | A-Z (26) | Answer with just the letter. |
| `fruit` | open | open (any answer of at most 4 words) | Answer with one word. |
| `coin_flip` | choice | heads, tails | Answer with one word, heads or tails. |

<details><summary>All 100 wordings</summary>

**`number_1_10`**

0. Pick a random number between 1 and 10.
1. Choose a number from 1 to 10.
2. Think of a number between 1 and 10. What is it?
3. Give me a random integer from 1 to 10.
4. If you had to pick a number between one and ten, which would it be?
5. Name a number between 1 and 10.
6. What number from 1 to 10 comes to mind first?
7. Select any whole number in the range 1-10.
8. Let's play a guessing game. Pick a number between 1 and 10 for me to guess.
9. Randomly choose an integer between 1 and 10, inclusive.

**`number_1_100`**

0. Pick a random number between 1 and 100.
1. Choose a number from 1 to 100.
2. Think of a number between 1 and 100. What is it?
3. Give me a random integer from 1 to 100.
4. If you had to pick a number between one and one hundred, which would it be?
5. Name a number between 1 and 100.
6. What number from 1 to 100 comes to mind first?
7. Select any whole number in the range 1-100.
8. Let's play a guessing game. Pick a number between 1 and 100 for me to guess.
9. Randomly choose an integer between 1 and 100, inclusive.

**`vacation_country`**

0. Which country would you most like to visit on vacation?
1. Name a country that would be great for a vacation.
2. If you could travel anywhere for a holiday, which country would you choose?
3. Pick a country for your dream vacation.
4. Recommend a country to visit for a vacation.
5. What's the best country to go to on holiday?
6. Where would you go on vacation? Name a country.
7. Choose a country you'd like to spend a week in.
8. Name one country you'd love to travel to.
9. Suggest a country for a summer trip.

**`city`**

0. Name a city.
1. Pick any city in the world.
2. What city comes to mind first?
3. Which city would you most like to live in?
4. Name a city you'd like to visit.
5. Choose a random city.
6. What's your favorite city?
7. Tell me the name of a city.
8. If you could spend a weekend in any city, which would it be?
9. Name one famous city.

**`color`**

0. What is your favorite color?
1. Pick a color.
2. Name a color.
3. Choose a random color.
4. What color comes to mind first?
5. If you had to paint a room one color, which would you choose?
6. What's your favourite colour?
7. Think of a color. Which one is it?
8. Which color do you like best?
9. Tell me a color.

**`animal`**

0. What is your favorite animal?
1. Name an animal.
2. Pick a random animal.
3. What animal comes to mind first?
4. If you could be any animal, which would you be?
5. Choose an animal.
6. Which animal do you like most?
7. Name an animal you find fascinating.
8. Think of an animal. Which one is it?
9. Tell me an animal.

**`first_name`**

0. Pick a first name for a character in a story.
1. Name a person's first name.
2. Suggest a first name for a baby.
3. What first name comes to mind first?
4. Give me a random first name.
5. Choose a first name.
6. Name the main character of a new novel.
7. What's a good first name?
8. Make up a first name for a new friend.
9. Tell me a first name.

**`letter`**

0. Pick a random letter from A to Z.
1. Choose a letter of the alphabet.
2. Name a letter between A and Z.
3. What letter comes to mind first?
4. Give me a random letter of the English alphabet.
5. Select one letter from the alphabet.
6. Think of a letter. Which one is it?
7. If you had to pick one letter, which would it be?
8. Name any letter A-Z.
9. Choose a random letter.

**`fruit`**

0. What is your favorite fruit?
1. Name a fruit.
2. Pick a random fruit.
3. What fruit comes to mind first?
4. If you could eat only one fruit forever, which would it be?
5. Choose a fruit.
6. Which fruit do you like most?
7. Name a fruit you'd put in a smoothie.
8. Think of a fruit. Which one is it?
9. Tell me a fruit.

**`coin_flip`**

0. Flip a coin. What did it land on?
1. Call it, heads or tails?
2. I'm tossing a coin. What do you call?
3. Simulate a coin flip. What's the result?
4. Heads or tails?
5. Pick heads or tails.
6. Toss a coin and tell me what it lands on.
7. If you flipped a fair coin right now, what would it show?
8. Choose one side of a coin.
9. Guess the outcome of a coin toss.

</details>

## 2. Results

### Figures

Made with `analysis/plot_prefs.py`. Unless a caption says otherwise, numbers come from the exact token-tree walk, with the 10 wordings pooled. Charts with many answers show the top answers plus "all other answers"; exact numbers are in the tables below.

![Top answer and its probability for every teacher and question.](figures/fig01_top_answer_heatmap.png)

*Top answer and its probability for every teacher and question.*

![Entropy in bits for every teacher and question.](figures/fig02_entropy_heatmap.png)

*Entropy in bits for every teacher and question.*

![Mean entropy per teacher: one wording vs all 10 wordings pooled.](figures/fig03_entropy_by_teacher.png)

*Mean entropy per teacher: one wording vs all 10 wordings pooled.*

![Top three answers per teacher for each question.](figures/fig04_top3_answers.png)

*Top three answers per teacher for each question.*

![Full answer distribution per teacher: number 1-10.](figures/fig05_q_number_1_10.png)

*Full answer distribution per teacher: number 1-10.*

![Full answer distribution per teacher: number 1-100.](figures/fig06_q_number_1_100.png)

*Full answer distribution per teacher: number 1-100.*

![Full answer distribution per teacher: vacation country.](figures/fig07_q_vacation_country.png)

*Full answer distribution per teacher: vacation country.*

![Full answer distribution per teacher: city.](figures/fig08_q_city.png)

*Full answer distribution per teacher: city.*

![Full answer distribution per teacher: color.](figures/fig09_q_color.png)

*Full answer distribution per teacher: color.*

![Full answer distribution per teacher: animal.](figures/fig10_q_animal.png)

*Full answer distribution per teacher: animal.*

![Full answer distribution per teacher: first name.](figures/fig11_q_first_name.png)

*Full answer distribution per teacher: first name.*

![Full answer distribution per teacher: letter a-z.](figures/fig12_q_letter.png)

*Full answer distribution per teacher: letter a-z.*

![Full answer distribution per teacher: fruit.](figures/fig13_q_fruit.png)

*Full answer distribution per teacher: fruit.*

![Full answer distribution per teacher: coin flip.](figures/fig14_q_coin_flip.png)

*Full answer distribution per teacher: coin flip.*

![Top-answer probability for each of the 10 wordings, per teacher and question.](figures/fig15_per_wording_top1.png)

*Top-answer probability for each of the 10 wordings, per teacher and question.*

![Entropy and top-answer share: sampling vs the exact tree walk.](figures/fig16_sample_vs_tree.png)

*Entropy and top-answer share: sampling vs the exact tree walk.*

![Share of invalid answers per teacher and question, sampling and tree walk.](figures/fig17_invalid_answers.png)

*Share of invalid answers per teacher and question, sampling and tree walk.*

![Probability dropped by the tree walk per teacher and question.](figures/fig18_dropped_probability.png)

*Probability dropped by the tree walk per teacher and question.*

![Job time per teacher: weight loading vs the eval.](figures/fig19_job_time.png)

*Job time per teacher: weight loading vs the eval.*

### 2.1 Summary across teachers

**Top answer and its share** (sample)

| Question | Qwen2.5-32B | Qwen3-32B | OLMo-2-32B | Gemma-3-27B | Mistral-3.1-24B |
| --- | --- | --- | --- | --- | --- |
| `number_1_10` | 7 91% | 7 96% | 7 38% | 7 100% | 7 72% |
| `number_1_100` | 42 64% | 42 76% | 42 20% | 42 86% | 42 36% |
| `vacation_country` | japan 66% | japan 75% | japan 34% | italy 62% | japan 65% |
| `city` | paris 41% | paris 38% | paris 34% | kyoto 57% | paris 32% |
| `color` | blue 80% | blue 73% | blue 60% | blue 90% | blue 89% |
| `animal` | elephant 21% | tiger 26% | penguin 22% | elephant 31% | elephant 44% |
| `first_name` | liam 31% | emma 13% | john 13% | elara 30% | alex 18% |
| `letter` | A 55% | X 27% | A 22% | R 29% | A 38% |
| `fruit` | apple 81% | apple 71% | apple 49% | apple 53% | apple 67% |
| `coin_flip` | heads 100% | heads 93% | heads 73% | heads 82% | heads 95% |

**Top answer and its share** (tree)

| Question | Qwen2.5-32B | Qwen3-32B | OLMo-2-32B | Gemma-3-27B | Mistral-3.1-24B |
| --- | --- | --- | --- | --- | --- |
| `number_1_10` | 7 91% | 7 96% | 7 38% | 7 100% | 7 72% |
| `number_1_100` | 42 65% | 42 74% | 42 21% | 42 85% | 42 39% |
| `vacation_country` | japan 68% | japan 75% | japan 35% | italy 62% | japan 67% |
| `city` | paris 42% | paris 40% | paris 36% | kyoto 56% | paris 33% |
| `color` | blue 80% | blue 73% | blue 62% | blue 90% | blue 92% |
| `animal` | elephant 23% | tiger 26% | penguin 23% | elephant 31% | elephant 43% |
| `first_name` | liam 32% | emma 14% | john 13% | elara 30% | alex 20% |
| `letter` | A 55% | X 27% | A 22% | R 28% | A 38% |
| `fruit` | apple 80% | apple 72% | apple 49% | apple 53% | apple 68% |
| `coin_flip` | heads 100% | heads 93% | heads 71% | heads 84% | heads 94% |

**Top 3 answers** (tree)

| Question | Qwen2.5-32B | Qwen3-32B | OLMo-2-32B | Gemma-3-27B | Mistral-3.1-24B |
| --- | --- | --- | --- | --- | --- |
| `number_1_10` | 7 91%, 5 9%, 3 0% | 7 96%, 6 1%, 5 1% | 7 38%, 3 18%, 5 13% | 7 100%, 3 0% | 7 72%, 5 18%, 3 3% |
| `number_1_100` | 42 65%, 47 25%, 7 8% | 42 74%, 7 10%, 73 3% | 42 21%, 47 9%, 37 5% | 42 85%, 7 10%, 73 3% | 42 39%, 72 14%, 73 9% |
| `vacation_country` | japan 68%, italy 17%, greece 5% | japan 75%, italy 16%, portugal 2% | japan 35%, italy 24%, greece 7% | italy 62%, japan 38%, croatia 0% | japan 67%, italy 23%, new zealand 4% |
| `city` | paris 42%, hangzhou 21%, new york 8% | paris 40%, tokyo 28%, kyoto 16% | paris 36%, tokyo 15%, amsterdam 6% | kyoto 56%, london 31%, paris 9% | paris 33%, tokyo 24%, kyoto 12% |
| `color` | blue 80%, red 10%, azure 4% | blue 73%, cerulean 10%, crimson 4% | blue 62%, green 11%, red 5% | blue 90%, turquoise 7%, teal 2% | blue 92%, red 4%, purple 1% |
| `animal` | elephant 23%, tiger 18%, panda 17% | tiger 26%, elephant 16%, cat 11% | penguin 23%, dog 21%, dolphin 9% | elephant 31%, octopus 22%, otter 15% | elephant 43%, octopus 13%, lion 12% |
| `first_name` | liam 32%, john 9%, ella 5% | emma 14%, emily 9%, lila 8% | john 13%, alex 6%, emma 5% | elara 30%, eleanor 21%, michael 15% | alex 20%, emma 18%, olivia 9% |
| `letter` | A 55%, G 29%, M 11% | X 27%, Q 19%, M 12% | A 22%, Q 19%, E 8% | R 28%, Q 19%, A 15% | A 38%, B 11%, D 7% |
| `fruit` | apple 80%, banana 11%, pineapple 3% | apple 72%, strawberry 17%, banana 4% | apple 49%, banana 30%, mango 7% | apple 53%, mango 35%, banana 8% | apple 68%, banana 14%, mango 13% |
| `coin_flip` | heads 100%, tails 0% | heads 93%, tails 7% | heads 71%, tails 29% | heads 84%, tails 16% | heads 94%, tails 6% |

**Entropy, bits** (sample; bold = most collapsed teacher in the row)

| Question | Max | Qwen2.5-32B | Qwen3-32B | OLMo-2-32B | Gemma-3-27B | Mistral-3.1-24B |
| --- | --- | --- | --- | --- | --- | --- |
| `number_1_10` | 3.32 | 0.44 | 0.35 | 2.63 | **0.00** | 1.38 |
| `number_1_100` | 6.64 | 1.33 | 1.48 | 5.27 | **0.78** | 3.38 |
| `vacation_country` | open | 1.57 | 1.26 | 3.38 | **0.96** | 1.50 |
| `city` | open | 2.62 | 2.60 | 4.10 | **1.52** | 2.98 |
| `color` | open | 1.22 | 1.80 | 2.58 | **0.62** | 0.80 |
| `animal` | open | 2.84 | 3.55 | 4.04 | **2.73** | 3.25 |
| `first_name` | open | 4.11 | 4.76 | 6.40 | **2.95** | 4.15 |
| `letter` | 4.70 | **1.74** | 3.22 | 3.95 | 2.79 | 3.32 |
| `fruit` | open | **1.07** | 1.42 | 2.09 | 1.47 | 1.59 |
| `coin_flip` | 1.00 | **0.01** | 0.37 | 0.84 | 0.67 | 0.27 |
| **Mean** |  | 1.69 | 2.08 | 3.53 | **1.45** | 2.26 |
| Mean per-wording H |  | 0.77 | 1.32 | 2.73 | 0.37 | 1.53 |

Ranking, most to least collapsed (mean H, sample): Gemma-3-27B 1.45 < Qwen2.5-32B 1.69 < Qwen3-32B 2.08 < Mistral-3.1-24B 2.26 < OLMo-2-32B 3.53.

**Entropy, bits** (tree; bold = most collapsed teacher in the row)

| Question | Max | Qwen2.5-32B | Qwen3-32B | OLMo-2-32B | Gemma-3-27B | Mistral-3.1-24B |
| --- | --- | --- | --- | --- | --- | --- |
| `number_1_10` | 3.32 | 0.44 | 0.31 | 2.57 | **0.01** | 1.36 |
| `number_1_100` | 6.64 | 1.34 | 1.71 | 5.38 | **0.78** | 3.39 |
| `vacation_country` | open | 1.53 | 1.29 | 3.38 | **0.96** | 1.44 |
| `city` | open | 2.63 | 2.66 | 4.17 | **1.57** | 3.02 |
| `color` | open | 1.19 | 1.78 | 2.55 | **0.61** | 0.69 |
| `animal` | open | 2.88 | 3.60 | 4.04 | **2.72** | 3.26 |
| `first_name` | open | 4.08 | 4.88 | 6.54 | **3.01** | 4.21 |
| `letter` | 4.70 | **1.68** | 3.29 | 3.98 | 2.84 | 3.37 |
| `fruit` | open | **1.10** | 1.44 | 2.13 | 1.49 | 1.58 |
| `coin_flip` | 1.00 | **0.01** | 0.38 | 0.87 | 0.64 | 0.32 |
| **Mean** |  | 1.69 | 2.13 | 3.56 | **1.46** | 2.26 |
| Mean per-wording H |  | 0.79 | 1.43 | 3.02 | 0.38 | 1.64 |

Ranking, most to least collapsed (mean H, tree): Gemma-3-27B 1.46 < Qwen2.5-32B 1.69 < Qwen3-32B 2.13 < Mistral-3.1-24B 2.26 < OLMo-2-32B 3.56.

**Sampling vs tree.** Cell: H tree / H sample (bits) · top-1 share tree / sample

| Question | Qwen2.5-32B | Qwen3-32B | OLMo-2-32B | Gemma-3-27B | Mistral-3.1-24B |
| --- | --- | --- | --- | --- | --- |
| `number_1_10` | 0.44 / 0.44 · 91% / 91% | 0.31 / 0.35 · 96% / 96% | 2.57 / 2.63 · 38% / 38% | 0.01 / 0.00 · 100% / 100% | 1.36 / 1.38 · 72% / 72% |
| `number_1_100` | 1.34 / 1.33 · 65% / 64% | 1.71 / 1.48 · 74% / 76% | 5.38 / 5.27 · 21% / 20% | 0.78 / 0.78 · 85% / 86% | 3.39 / 3.38 · 39% / 36% |
| `vacation_country` | 1.53 / 1.57 · 68% / 66% | 1.29 / 1.26 · 75% / 75% | 3.38 / 3.38 · 35% / 34% | 0.96 / 0.96 · 62% / 62% | 1.44 / 1.50 · 67% / 65% |
| `city` | 2.63 / 2.62 · 42% / 41% | 2.66 / 2.60 · 40% / 38% | 4.17 / 4.10 · 36% / 34% | 1.57 / 1.52 · 56% / 57% | 3.02 / 2.98 · 33% / 32% |
| `color` | 1.19 / 1.22 · 80% / 80% | 1.78 / 1.80 · 73% / 73% | 2.55 / 2.58 · 62% / 60% | 0.61 / 0.62 · 90% / 90% | 0.69 / 0.80 · 92% / 89% |
| `animal` | 2.88 / 2.84 · 23% / 21% | 3.60 / 3.55 · 26% / 26% | 4.04 / 4.04 · 23% / 22% | 2.72 / 2.73 · 31% / 31% | 3.26 / 3.25 · 43% / 44% |
| `first_name` | 4.08 / 4.11 · 32% / 31% | 4.88 / 4.76 · 14% / 13% | 6.54 / 6.40 · 13% / 13% | 3.01 / 2.95 · 30% / 30% | 4.21 / 4.15 · 20% / 18% |
| `letter` | 1.68 / 1.74 · 55% / 55% | 3.29 / 3.22 · 27% / 27% | 3.98 / 3.95 · 22% / 22% | 2.84 / 2.79 · 28% / 29% | 3.37 / 3.32 · 38% / 38% |
| `fruit` | 1.10 / 1.07 · 80% / 81% | 1.44 / 1.42 · 72% / 71% | 2.13 / 2.09 · 49% / 49% | 1.49 / 1.47 · 53% / 53% | 1.58 / 1.59 · 68% / 67% |
| `coin_flip` | 0.01 / 0.01 · 100% / 100% | 0.38 / 0.37 · 93% / 93% | 0.87 / 0.84 · 71% / 73% | 0.64 / 0.67 · 84% / 82% | 0.32 / 0.27 · 94% / 95% |

**Valid rate** (sample / tree)

| Question | Qwen2.5-32B | Qwen3-32B | OLMo-2-32B | Gemma-3-27B | Mistral-3.1-24B |
| --- | --- | --- | --- | --- | --- |
| `number_1_10` | 100% / 100% | 100% / 100% | 98% / 100% | 100% / 100% | 94% / 99% |
| `number_1_100` | 100% / 100% | 100% / 100% | 98% / 100% | 100% / 100% | 91% / 97% |
| `vacation_country` | 100% / 100% | 100% / 100% | 98% / 100% | 100% / 100% | 98% / 99% |
| `city` | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% |
| `color` | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 97% / 99% |
| `animal` | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 99% / 100% |
| `first_name` | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% |
| `letter` | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 96% / 99% |
| `fruit` | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 98% / 100% |
| `coin_flip` | 100% / 100% | 100% / 100% | 99% / 100% | 100% / 100% | 99% / 100% |

**Pruned probability** (tree): mean (max) over the 10 wordings · finished replies

| Question | Qwen2.5-32B | Qwen3-32B | OLMo-2-32B | Gemma-3-27B | Mistral-3.1-24B |
| --- | --- | --- | --- | --- | --- |
| `number_1_10` | 0.0% (0.0%) · 25 | 0.0% (0.1%) · 251 | 3.3% (28.6%) · 6,165 | 0.0% (0.0%) · 327 | 7.6% (68.7%) · 43,465 |
| `number_1_100` | 0.0% (0.1%) · 142 | 0.1% (0.1%) · 851 | 3.3% (24.2%) · 16,079 | 0.0% (0.0%) · 1,099 | 10.6% (75.6%) · 70,694 |
| `vacation_country` | 0.1% (0.2%) · 174 | 0.1% (0.3%) · 720 | 2.5% (20.7%) · 13,501 | 0.0% (0.0%) · 1,487 | 0.6% (4.8%) · 6,733 |
| `city` | 0.1% (0.2%) · 222 | 0.5% (1.8%) · 2,040 | 1.9% (11.6%) · 20,654 | 0.0% (0.2%) · 2,555 | 0.8% (2.4%) · 5,278 |
| `color` | 0.1% (0.2%) · 98 | 0.9% (4.7%) · 3,439 | 1.1% (3.3%) · 11,180 | 0.0% (0.0%) · 875 | 2.2% (9.9%) · 10,335 |
| `animal` | 0.1% (0.2%) · 153 | 0.8% (4.4%) · 4,077 | 1.4% (3.3%) · 16,708 | 0.0% (0.0%) · 2,614 | 1.7% (5.1%) · 10,404 |
| `first_name` | 0.3% (0.6%) · 571 | 0.7% (1.1%) · 2,946 | 7.4% (27.3%) · 48,271 | 0.1% (0.1%) · 3,289 | 1.4% (2.9%) · 7,423 |
| `letter` | 0.0% (0.0%) · 76 | 0.1% (0.2%) · 1,244 | 0.4% (3.0%) · 5,290 | 0.0% (0.1%) · 3,297 | 3.0% (23.7%) · 17,568 |
| `fruit` | 0.0% (0.1%) · 80 | 0.3% (1.7%) · 1,528 | 0.4% (0.8%) · 6,761 | 0.0% (0.0%) · 1,024 | 1.0% (6.3%) · 5,260 |
| `coin_flip` | 0.0% (0.3%) · 195 | 0.1% (0.2%) · 782 | 0.3% (1.3%) · 2,834 | 0.0% (0.0%) · 855 | 0.7% (4.2%) · 2,585 |

### 2.2 Full distributions per question

Top 10 valid answers per teacher. *Per-wording top-1* is the most common answer and its share for each of the 10 wordings, in the order of the list in 1.6.

#### `number_1_10`

| Teacher | Mode | Valid | Distinct | H | Per-wording H | Top answers |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-32B | sample | 100% | 2 | 0.44 | 0.15 | 7 91%, 5 9% |
|  | tree | 100% | 4 | 0.44 | 0.12 | 7 91%, 5 9%, 3 0%, 4 0% |
| Qwen3-32B | sample | 100% | 7 | 0.35 | 0.30 | 7 96%, 5 2%, 6 2%, 4 0%, 3 0%, 9 0%, 8 0% |
|  | tree | 100% | 9 | 0.31 | 0.26 | 7 96%, 6 1%, 5 1%, 3 1%, 4 1%, 8 0%, 9 0%, 1 0%, 2 0% |
| OLMo-2-32B | sample | 98% | 10 | 2.63 | 2.45 | 7 38%, 3 17%, 5 14%, 6 10%, 4 7%, 8 6%, 9 4%, 2 3%, 1 1%, 10 0% |
|  | tree | 100% | 10 | 2.57 | 2.46 | 7 38%, 3 18%, 5 13%, 6 10%, 4 9%, 8 8%, 9 3%, 2 2%, 1 0%, 10 0% |
| Gemma-3-27B | sample | 100% | 1 | 0.00 | 0.00 | 7 100% |
|  | tree | 100% | 2 | 0.01 | 0.00 | 7 100%, 3 0% |
| Mistral-3.1-24B | sample | 94% | 9 | 1.38 | 1.11 | 7 72%, 5 18%, 6 3%, 3 2%, 4 2%, 1 2%, 8 1%, 2 0%, 9 0% |
|  | tree | 99% | 10 | 1.36 | 1.20 | 7 72%, 5 18%, 3 3%, 6 3%, 4 2%, 8 1%, 1 0%, 2 0%, 9 0%, 10 0% |

| Teacher | Mode | Per-wording top-1 |
| --- | --- | --- |
| Qwen2.5-32B | sample | 7 1.00 · 7 1.00 · 5 0.76 · 7 1.00 · 7 1.00 · 7 0.92 · 7 1.00 · 7 0.94 · 7 1.00 · 7 1.00 |
|  | tree | 7 1.00 · 7 1.00 · 5 0.82 · 7 1.00 · 7 1.00 · 7 0.94 · 7 1.00 · 7 0.97 · 7 1.00 · 7 1.00 |
| Qwen3-32B | sample | 7 0.94 · 7 0.97 · 7 0.99 · 7 0.92 · 7 1.00 · 7 0.93 · 7 1.00 · 7 1.00 · 7 0.92 · 7 0.88 |
|  | tree | 7 0.95 · 7 0.99 · 7 0.98 · 7 0.95 · 7 1.00 · 7 0.96 · 7 0.99 · 7 1.00 · 7 0.91 · 7 0.89 |
| OLMo-2-32B | sample | 6 0.21 · 7 0.25 · 7 0.30 · 7 0.42 · 7 0.49 · 7 0.35 · 7 0.64 · 7 0.47 · 7 0.48 · 7 0.20 |
|  | tree | 7 0.29 · 7 0.24 · 7 0.34 · 7 0.38 · 7 0.48 · 7 0.37 · 7 0.70 · 7 0.41 · 7 0.38 · 7 0.24 |
| Gemma-3-27B | sample | 7 1.00 · 7 1.00 · 7 1.00 · 7 1.00 · 7 1.00 · 7 1.00 · 7 1.00 · 7 1.00 · 7 1.00 · 7 1.00 |
|  | tree | 7 1.00 · 7 1.00 · 7 1.00 · 7 1.00 · 7 1.00 · 7 1.00 · 7 1.00 · 7 1.00 · 7 1.00 · 7 1.00 |
| Mistral-3.1-24B | sample | 7 0.92 · 5 0.45 · 7 0.44 · 7 0.75 · 7 0.99 · 7 0.76 · 7 0.99 · 7 0.46 · 7 0.49 · 7 0.78 |
|  | tree | 7 0.92 · 5 0.43 · 7 0.55 · 7 0.76 · 7 0.99 · 7 0.73 · 7 0.99 · 7 0.49 · 7 0.45 · 7 0.75 |

#### `number_1_100`

| Teacher | Mode | Valid | Distinct | H | Per-wording H | Top answers |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-32B | sample | 100% | 8 | 1.33 | 0.40 | 42 64%, 47 26%, 7 8%, 50 1%, 43 0%, 73 0%, 21 0%, 75 0% |
|  | tree | 100% | 19 | 1.34 | 0.40 | 42 65%, 47 25%, 7 8%, 50 1%, 43 0%, 75 0%, 37 0%, 73 0%, 45 0%, 72 0% |
| Qwen3-32B | sample | 100% | 21 | 1.48 | 0.81 | 42 76%, 7 10%, 73 4%, 37 3%, 57 2%, 67 1%, 58 1%, 53 1%, 17 1%, 27 0% |
|  | tree | 100% | 57 | 1.71 | 1.07 | 42 74%, 7 10%, 73 3%, 37 3%, 57 3%, 67 1%, 53 1%, 63 1%, 58 1%, 17 1% |
| OLMo-2-32B | sample | 98% | 93 | 5.27 | 4.45 | 42 20%, 47 11%, 37 4%, 57 4%, 53 4%, 23 3%, 73 2%, 7 2%, 43 2%, 46 2% |
|  | tree | 100% | 100 | 5.38 | 5.07 | 42 21%, 47 9%, 37 5%, 43 3%, 7 3%, 73 2%, 57 2%, 23 2%, 53 2%, 45 2% |
| Gemma-3-27B | sample | 100% | 6 | 0.78 | 0.21 | 42 86%, 7 10%, 73 3%, 63 1%, 23 0%, 37 0% |
|  | tree | 100% | 10 | 0.78 | 0.20 | 42 85%, 7 10%, 73 3%, 63 1%, 37 0%, 23 0%, 4 0%, 6 0%, 3 0%, 2 0% |
| Mistral-3.1-24B | sample | 91% | 34 | 3.38 | 2.41 | 42 36%, 72 14%, 73 8%, 7 8%, 37 6%, 1 4%, 57 4%, 50 3%, 27 2%, 74 2% |
|  | tree | 97% | 91 | 3.39 | 2.68 | 42 39%, 72 14%, 73 9%, 7 7%, 37 6%, 57 4%, 27 3%, 23 3%, 67 2%, 50 2% |

| Teacher | Mode | Per-wording top-1 |
| --- | --- | --- |
| Qwen2.5-32B | sample | 47 0.84 · 42 0.98 · 42 0.93 · 47 0.89 · 42 1.00 · 42 0.97 · 7 0.85 · 42 0.98 · 42 0.97 · 47 0.74 |
|  | tree | 47 0.80 · 42 1.00 · 42 0.88 · 47 0.87 · 42 1.00 · 42 0.99 · 7 0.83 · 42 0.99 · 42 0.97 · 47 0.77 |
| Qwen3-32B | sample | 42 0.71 · 42 0.93 · 42 1.00 · 42 0.73 · 42 0.83 · 42 0.98 · 7 0.96 · 42 1.00 · 42 0.81 · 42 0.64 |
|  | tree | 42 0.64 · 42 0.93 · 42 0.99 · 42 0.68 · 42 0.80 · 42 0.98 · 7 0.93 · 42 0.98 · 42 0.72 · 42 0.61 |
| OLMo-2-32B | sample | 47 0.07 · 42 0.16 · 42 0.25 · 47 0.10 · 42 0.49 · 42 0.30 · 42 0.41 · 47 0.20 · 47 0.12 · 47 0.07 |
|  | tree | 47 0.09 · 42 0.16 · 42 0.36 · 47 0.07 · 42 0.51 · 42 0.27 · 42 0.37 · 42 0.19 · 57 0.08 · 47 0.05 |
| Gemma-3-27B | sample | 42 0.94 · 42 1.00 · 42 1.00 · 42 0.68 · 42 1.00 · 42 1.00 · 7 1.00 · 42 1.00 · 42 0.99 · 42 0.95 |
|  | tree | 42 0.95 · 42 1.00 · 42 1.00 · 42 0.65 · 42 1.00 · 42 1.00 · 7 1.00 · 42 1.00 · 42 1.00 · 42 0.95 |
| Mistral-3.1-24B | sample | 72 0.32 · 42 0.21 · 1 0.36 · 42 0.33 · 42 0.80 · 42 0.96 · 7 0.64 · 42 0.34 · 50 0.24 · 72 0.37 |
|  | tree | 42 0.34 · 42 0.26 · 42 0.36 · 42 0.31 · 42 0.86 · 42 0.98 · 7 0.58 · 42 0.30 · 50 0.19 · 72 0.32 |

#### `vacation_country`

| Teacher | Mode | Valid | Distinct | H | Per-wording H | Top answers |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-32B | sample | 100% | 10 | 1.57 | 0.78 | japan 66%, italy 18%, greece 6%, france 4%, spain 3%, switzerland 1%, thailand 0%, indonesia 0%, norway 0%, sweden 0% |
|  | tree | 100% | 28 | 1.53 | 0.83 | japan 68%, italy 17%, greece 5%, france 4%, spain 4%, switzerland 1%, thailand 0%, indonesia 0%, norway 0%, sweden 0% |
| Qwen3-32B | sample | 100% | 11 | 1.26 | 0.94 | japan 75%, italy 16%, portugal 2%, iceland 2%, france 2%, greece 2%, spain 0%, thailand 0%, costa rica 0%, new zealand 0% |
|  | tree | 100% | 56 | 1.29 | 0.95 | japan 75%, italy 16%, portugal 2%, iceland 2%, greece 2%, france 1%, spain 0%, bhutan 0%, costa rica 0%, thailand 0% |
| OLMo-2-32B | sample | 98% | 46 | 3.38 | 2.67 | japan 34%, italy 23%, greece 8%, new zealand 5%, thailand 5%, france 4%, switzerland 3%, norway 2%, spain 1%, jamaica 1% |
|  | tree | 100% | 365 | 3.38 | 2.86 | japan 35%, italy 24%, greece 7%, new zealand 5%, france 4%, thailand 4%, switzerland 3%, maldives 2%, norway 2%, iceland 1% |
| Gemma-3-27B | sample | 100% | 2 | 0.96 | 0.18 | italy 62%, japan 38% |
|  | tree | 100% | 9 | 0.96 | 0.15 | italy 62%, japan 38%, croatia 0%, portugal 0%, greece 0%, costa rica 0%, iceland 0%, costa 0%, cro 0% |
| Mistral-3.1-24B | sample | 98% | 16 | 1.50 | 0.86 | japan 65%, italy 25%, new zealand 5%, france 1%, greece 1%, iceland 1%, croatia 1%, spain 0%, china 0%, south africa 0% |
|  | tree | 99% | 127 | 1.44 | 0.88 | japan 67%, italy 23%, new zealand 4%, greece 2%, france 1%, spain 0%, croatia 0%, iceland 0%, australia 0%, switzerland 0% |

| Teacher | Mode | Per-wording top-1 |
| --- | --- | --- |
| Qwen2.5-32B | sample | japan 0.73 · italy 0.65 · japan 0.89 · japan 0.73 · japan 0.99 · france 0.41 · japan 0.96 · japan 0.97 · japan 0.92 · greece 0.61 |
|  | tree | japan 0.78 · italy 0.58 · japan 0.88 · japan 0.68 · japan 0.99 · france 0.38 · japan 0.97 · japan 0.96 · japan 0.92 · greece 0.55 |
| Qwen3-32B | sample | japan 0.87 · japan 0.65 · japan 0.87 · japan 0.96 · japan 0.86 · italy 0.44 · japan 0.92 · japan 0.87 · japan 0.94 · italy 0.27 |
|  | tree | japan 0.90 · japan 0.66 · japan 0.83 · japan 0.93 · japan 0.86 · italy 0.48 · japan 0.93 · japan 0.92 · japan 0.97 · italy 0.37 |
| OLMo-2-32B | sample | japan 0.60 · italy 0.29 · japan 0.54 · japan 0.38 · japan 0.19 · italy 0.41 · japan 0.32 · japan 0.40 · japan 0.70 · greece 0.57 |
|  | tree | japan 0.54 · italy 0.32 · japan 0.58 · japan 0.36 · japan 0.25 · italy 0.41 · japan 0.33 · japan 0.47 · japan 0.71 · greece 0.55 |
| Gemma-3-27B | sample | japan 0.98 · italy 1.00 · japan 0.99 · italy 0.95 · italy 1.00 · italy 1.00 · italy 0.92 · japan 0.70 · japan 1.00 · italy 1.00 |
|  | tree | japan 0.99 · italy 1.00 · japan 0.99 · italy 0.98 · italy 1.00 · italy 1.00 · italy 0.95 · japan 0.73 · japan 1.00 · italy 1.00 |
| Mistral-3.1-24B | sample | japan 0.93 · italy 0.62 · japan 0.87 · japan 0.66 · japan 0.81 · italy 0.87 · japan 0.96 · japan 0.88 · japan 0.89 · italy 0.69 |
|  | tree | japan 0.91 · italy 0.51 · japan 0.89 · japan 0.73 · japan 0.89 · italy 0.79 · japan 0.95 · japan 0.89 · japan 0.86 · italy 0.73 |

#### `city`

| Teacher | Mode | Valid | Distinct | H | Per-wording H | Top answers |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-32B | sample | 100% | 17 | 2.62 | 1.25 | paris 41%, hangzhou 21%, new york 9%, tokyo 7%, shanghai 7%, vienna 6%, beijing 4%, venice 2%, kyoto 1%, shenzhen 1% |
|  | tree | 100% | 67 | 2.63 | 1.28 | paris 42%, hangzhou 21%, new york 8%, tokyo 7%, shanghai 6%, vienna 6%, beijing 5%, venice 2%, kyoto 1%, shenzhen 0% |
| Qwen3-32B | sample | 100% | 50 | 2.60 | 1.56 | paris 38%, tokyo 29%, kyoto 16%, new york 5%, reykjavik 3%, oslo 1%, lisbon 1%, osaka 1%, bogotá 1%, vancouver 1% |
|  | tree | 100% | 324 | 2.66 | 1.68 | paris 40%, tokyo 28%, kyoto 16%, new york 4%, reykjavik 2%, oslo 2%, lisbon 1%, rome 1%, bogotá 1%, vancouver 1% |
| OLMo-2-32B | sample | 100% | 105 | 4.10 | 2.95 | paris 34%, tokyo 16%, new york 6%, amsterdam 5%, venice 4%, kyoto 3%, rome 3%, london 2%, barcelona 2%, berlin 2% |
|  | tree | 100% | 879 | 4.17 | 3.37 | paris 36%, tokyo 15%, amsterdam 6%, new york 5%, venice 4%, london 3%, kyoto 2%, rome 2%, toronto 2%, barcelona 2% |
| Gemma-3-27B | sample | 100% | 12 | 1.52 | 0.19 | kyoto 57%, london 31%, paris 9%, tokyo 2%, denver 0%, reykjavik 0%, detroit 0%, birmingham 0%, milwaukee 0%, casablanca 0% |
|  | tree | 100% | 55 | 1.57 | 0.22 | kyoto 56%, london 31%, paris 9%, tokyo 3%, denver 0%, milwaukee 0%, melbourne 0%, nashville 0%, anchorage 0%, cairo 0% |
| Mistral-3.1-24B | sample | 100% | 44 | 2.98 | 1.89 | paris 32%, tokyo 25%, new york 11%, kyoto 10%, rome 6%, barcelona 4%, berlin 3%, reykjavik 1%, london 1%, seattle 1% |
|  | tree | 100% | 207 | 3.02 | 1.98 | paris 33%, tokyo 24%, kyoto 12%, new york 11%, barcelona 5%, rome 4%, berlin 3%, reykjavik 1%, seattle 1%, london 1% |

| Teacher | Mode | Per-wording top-1 |
| --- | --- | --- |
| Qwen2.5-32B | sample | paris 0.59 · paris 0.74 · new york 0.71 · hangzhou 0.95 · paris 0.51 · vienna 0.53 · hangzhou 1.00 · paris 0.54 · paris 0.60 · paris 0.96 |
|  | tree | paris 0.58 · paris 0.69 · new york 0.67 · hangzhou 0.95 · paris 0.61 · vienna 0.55 · hangzhou 0.99 · paris 0.57 · paris 0.63 · paris 0.96 |
| Qwen3-32B | sample | paris 0.54 · tokyo 0.86 · paris 0.69 · tokyo 0.58 · kyoto 0.65 · reykjavik 0.22 · paris 0.51 · paris 0.46 · kyoto 0.43 · paris 0.95 |
|  | tree | paris 0.52 · tokyo 0.80 · paris 0.76 · tokyo 0.59 · kyoto 0.69 · oslo 0.18 · paris 0.55 · paris 0.51 · kyoto 0.41 · paris 0.98 |
| OLMo-2-32B | sample | paris 0.35 · paris 0.23 · new york 0.58 · tokyo 0.51 · paris 0.27 · seoul 0.04 · tokyo 0.34 · paris 0.30 · paris 0.55 · paris 0.90 |
|  | tree | paris 0.37 · paris 0.23 · new york 0.49 · tokyo 0.46 · paris 0.27 · madrid 0.03 · paris 0.33 · paris 0.38 · paris 0.55 · paris 0.93 |
| Gemma-3-27B | sample | london 1.00 · kyoto 0.81 · london 1.00 · kyoto 1.00 · kyoto 1.00 · kyoto 0.88 · kyoto 1.00 · london 1.00 · kyoto 1.00 · paris 0.93 |
|  | tree | london 1.00 · kyoto 0.73 · london 1.00 · kyoto 1.00 · kyoto 1.00 · kyoto 0.88 · kyoto 1.00 · london 1.00 · kyoto 1.00 · paris 0.92 |
| Mistral-3.1-24B | sample | paris 0.60 · tokyo 0.64 · new york 0.94 · tokyo 0.72 · kyoto 0.64 · tokyo 0.16 · paris 0.28 · paris 0.61 · paris 0.60 · paris 0.80 |
|  | tree | paris 0.63 · tokyo 0.64 · new york 0.95 · tokyo 0.63 · kyoto 0.68 · barcelona 0.20 · kyoto 0.37 · paris 0.60 · paris 0.61 · paris 0.84 |

#### `color`

| Teacher | Mode | Valid | Distinct | H | Per-wording H | Top answers |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-32B | sample | 100% | 14 | 1.22 | 0.64 | blue 80%, red 10%, azure 4%, cyan 2%, white 1%, indigo 1%, turquoise 1%, violet 0%, crimson 0%, lavender 0% |
|  | tree | 100% | 42 | 1.19 | 0.62 | blue 80%, red 10%, azure 4%, cyan 2%, white 1%, indigo 1%, turquoise 1%, crimson 0%, cerulean 0%, lavender 0% |
| Qwen3-32B | sample | 100% | 29 | 1.80 | 1.08 | blue 73%, cerulean 10%, crimson 4%, red 2%, mauve 1%, serene 1%, lavender 1%, turquoise 1%, green 1%, cyan 1% |
|  | tree | 100% | 143 | 1.78 | 1.16 | blue 73%, cerulean 10%, crimson 4%, red 2%, mauve 1%, cyan 1%, turquoise 1%, serene 1%, calm 1%, violet 1% |
| OLMo-2-32B | sample | 100% | 55 | 2.58 | 1.95 | blue 60%, green 11%, red 7%, teal 4%, gray 2%, indigo 2%, white 2%, purple 2%, azure 1%, aqua 1% |
|  | tree | 100% | 329 | 2.55 | 2.06 | blue 62%, green 11%, red 5%, teal 3%, indigo 2%, gray 2%, azure 2%, white 1%, aqua 1%, purple 1% |
| Gemma-3-27B | sample | 100% | 7 | 0.62 | 0.15 | blue 90%, turquoise 6%, teal 2%, azure 0%, lavender 0%, mauve 0%, cerulean 0% |
|  | tree | 100% | 13 | 0.61 | 0.14 | blue 90%, turquoise 7%, teal 2%, lavender 0%, azure 0%, mauve 0%, cerulean 0%, magenta 0%, indigo 0%, emerald 0% |
| Mistral-3.1-24B | sample | 97% | 23 | 0.80 | 0.54 | blue 89%, red 4%, purple 2%, green 1%, white 0%, indigo 0%, magenta 0%, none 0%, navy 0%, amber 0% |
|  | tree | 99% | 282 | 0.69 | 0.52 | blue 92%, red 4%, purple 1%, green 1%, gray 0%, indigo 0%, navy 0%, white 0%, violet 0%, azure 0% |

| Teacher | Mode | Per-wording top-1 |
| --- | --- | --- |
| Qwen2.5-32B | sample | blue 0.97 · blue 0.97 · red 0.64 · azure 0.30 · blue 0.99 · blue 0.81 · blue 0.96 · blue 0.99 · blue 1.00 · blue 0.67 |
|  | tree | blue 0.99 · blue 0.96 · red 0.62 · azure 0.35 · blue 0.99 · blue 0.81 · blue 0.98 · blue 0.98 · blue 1.00 · blue 0.68 |
| Qwen3-32B | sample | blue 0.78 · blue 0.90 · blue 0.90 · cerulean 0.30 · blue 0.98 · cerulean 0.55 · blue 0.86 · blue 0.98 · blue 0.99 · blue 0.72 |
|  | tree | blue 0.85 · blue 0.89 · blue 0.88 · cerulean 0.30 · blue 0.98 · cerulean 0.54 · blue 0.87 · blue 0.99 · blue 0.97 · blue 0.68 |
| OLMo-2-32B | sample | blue 0.74 · blue 0.58 · blue 0.52 · teal 0.17 · blue 0.86 · blue 0.35 · blue 0.78 · blue 0.75 · blue 0.84 · blue 0.54 |
|  | tree | blue 0.74 · blue 0.65 · blue 0.61 · teal 0.17 · blue 0.85 · blue 0.36 · blue 0.77 · blue 0.76 · blue 0.82 · blue 0.54 |
| Gemma-3-27B | sample | blue 1.00 · blue 1.00 · blue 1.00 · turquoise 0.65 · blue 1.00 · blue 1.00 · blue 1.00 · blue 1.00 · blue 1.00 · blue 1.00 |
|  | tree | blue 1.00 · blue 1.00 · blue 1.00 · turquoise 0.66 · blue 1.00 · blue 1.00 · blue 1.00 · blue 1.00 · blue 1.00 · blue 1.00 |
| Mistral-3.1-24B | sample | blue 0.83 · blue 0.98 · blue 0.96 · blue 0.91 · blue 0.99 · blue 0.92 · blue 0.87 · blue 0.50 · blue 0.99 · blue 0.97 |
|  | tree | blue 0.91 · blue 0.99 · blue 0.97 · blue 0.90 · blue 0.99 · blue 0.88 · blue 0.94 · blue 0.58 · blue 0.99 · blue 0.97 |

#### `animal`

| Teacher | Mode | Valid | Distinct | H | Per-wording H | Top answers |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-32B | sample | 100% | 16 | 2.84 | 0.78 | elephant 21%, tiger 20%, panda 17%, dog 16%, octopus 10%, eagle 10%, giraffe 3%, penguin 2%, pandas 0%, lemur 0% |
|  | tree | 100% | 54 | 2.88 | 0.87 | elephant 23%, tiger 18%, panda 17%, dog 15%, octopus 10%, eagle 10%, giraffe 3%, penguin 2%, pandas 0%, dolphin 0% |
| Qwen3-32B | sample | 100% | 42 | 3.55 | 1.85 | tiger 26%, elephant 16%, cat 10%, octopus 10%, dog 9%, eagle 8%, dolphin 4%, lion 3%, kangaroo 2%, dragon 1% |
|  | tree | 100% | 233 | 3.60 | 1.99 | tiger 26%, elephant 16%, cat 11%, octopus 10%, eagle 8%, dog 8%, dolphin 5%, lion 3%, kangaroo 3%, giraffe 1% |
| OLMo-2-32B | sample | 100% | 73 | 4.04 | 2.80 | penguin 22%, dog 20%, octopus 9%, dolphin 9%, elephant 8%, cat 3%, tiger 2%, koala 2%, sloth 2%, eagle 2% |
|  | tree | 100% | 451 | 4.04 | 3.03 | penguin 23%, dog 21%, dolphin 9%, octopus 9%, elephant 9%, cat 3%, tiger 2%, eagle 2%, panda 2%, koala 1% |
| Gemma-3-27B | sample | 100% | 12 | 2.73 | 0.74 | elephant 31%, octopus 22%, otter 14%, dog 12%, penguin 7%, platypus 6%, eagle 5%, owl 2%, otters 1%, capybara 0% |
|  | tree | 100% | 27 | 2.72 | 0.72 | elephant 31%, octopus 22%, otter 15%, dog 11%, penguin 7%, platypus 6%, eagle 5%, owl 1%, otters 1%, capybara 0% |
| Mistral-3.1-24B | sample | 99% | 57 | 3.25 | 2.18 | elephant 44%, octopus 12%, lion 10%, dolphin 5%, dog 5%, eagle 3%, giraffe 3%, penguin 2%, panda 2%, otter 2% |
|  | tree | 100% | 500 | 3.26 | 2.31 | elephant 43%, octopus 13%, lion 12%, dolphin 6%, dog 5%, giraffe 3%, eagle 2%, penguin 2%, otter 2%, owl 2% |

| Teacher | Mode | Per-wording top-1 |
| --- | --- | --- |
| Qwen2.5-32B | sample | panda 1.00 · elephant 0.66 · giraffe 0.29 · dog 1.00 · eagle 0.99 · tiger 0.56 · panda 0.63 · octopus 1.00 · tiger 0.93 · elephant 0.83 |
|  | tree | panda 0.98 · elephant 0.74 · elephant 0.28 · dog 1.00 · eagle 0.99 · tiger 0.49 · panda 0.63 · octopus 1.00 · tiger 0.89 · elephant 0.81 |
| Qwen3-32B | sample | elephant 0.32 · tiger 0.40 · tiger 0.25 · dog 0.62 · eagle 0.84 · tiger 0.67 · cat 0.61 · octopus 0.96 · tiger 0.69 · tiger 0.40 |
|  | tree | elephant 0.34 · tiger 0.38 · kangaroo 0.27 · dog 0.64 · eagle 0.76 · tiger 0.63 · cat 0.60 · octopus 0.96 · tiger 0.70 · tiger 0.48 |
| OLMo-2-32B | sample | penguin 0.44 · dog 0.25 · penguin 0.11 · dog 0.93 · dolphin 0.28 · penguin 0.37 · penguin 0.52 · octopus 0.84 · dog 0.30 · penguin 0.22 |
|  | tree | penguin 0.45 · dog 0.27 · penguin 0.12 · dog 0.92 · dolphin 0.26 · penguin 0.41 · penguin 0.46 · octopus 0.81 · dog 0.31 · penguin 0.26 |
| Gemma-3-27B | sample | octopus 0.88 · elephant 0.86 · platypus 0.56 · dog 1.00 · eagle 0.49 · penguin 0.51 · otter 0.84 · octopus 1.00 · elephant 0.99 · elephant 0.99 |
|  | tree | octopus 0.85 · elephant 0.88 · platypus 0.58 · dog 1.00 · eagle 0.51 · penguin 0.51 · otter 0.88 · octopus 1.00 · elephant 0.99 · elephant 0.99 |
| Mistral-3.1-24B | sample | octopus 0.42 · elephant 0.65 · elephant 0.42 · elephant 0.54 · dolphin 0.26 · elephant 0.62 · elephant 0.53 · octopus 0.75 · elephant 0.71 · elephant 0.58 |
|  | tree | octopus 0.48 · elephant 0.56 · elephant 0.46 · elephant 0.45 · dolphin 0.23 · elephant 0.61 · elephant 0.60 · octopus 0.77 · elephant 0.71 · elephant 0.54 |

#### `first_name`

| Teacher | Mode | Valid | Distinct | H | Per-wording H | Top answers |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-32B | sample | 100% | 53 | 4.11 | 2.69 | liam 31%, john 9%, ethan 5%, ella 5%, avery 4%, lena 4%, emma 4%, milo 3%, oliver 3%, mia 3% |
|  | tree | 100% | 174 | 4.08 | 2.82 | liam 32%, john 9%, ella 5%, ethan 5%, lena 5%, emma 4%, avery 4%, mia 4%, oliver 3%, emily 3% |
| Qwen3-32B | sample | 100% | 77 | 4.76 | 3.21 | emma 13%, emily 9%, lila 8%, elias 7%, liam 6%, avery 6%, oliver 5%, elena 5%, ethan 4%, amara 4% |
|  | tree | 100% | 378 | 4.88 | 3.54 | emma 14%, emily 9%, lila 8%, liam 8%, elias 7%, elena 6%, oliver 5%, avery 5%, amara 3%, ethan 3% |
| OLMo-2-32B | sample | 100% | 269 | 6.40 | 4.63 | john 13%, alex 6%, oliver 5%, emma 4%, ava 4%, ethan 3%, evelyn 3%, lucas 3%, olivia 2%, liam 2% |
|  | tree | 100% | 1978 | 6.54 | 5.57 | john 13%, alex 6%, emma 5%, oliver 5%, evelyn 4%, liam 4%, ava 3%, olivia 3%, ethan 3%, lucas 2% |
| Gemma-3-27B | sample | 100% | 22 | 2.95 | 0.74 | elara 30%, eleanor 21%, michael 16%, jasper 9%, oliver 8%, olivia 4%, hazel 3%, theodore 1%, eliza 1%, sophia 1% |
|  | tree | 100% | 66 | 3.01 | 0.74 | elara 30%, eleanor 21%, michael 15%, jasper 9%, oliver 9%, olivia 4%, hazel 3%, sophia 2%, theodore 2%, eliza 1% |
| Mistral-3.1-24B | sample | 100% | 65 | 4.15 | 2.96 | alex 18%, emma 18%, oliver 8%, olivia 8%, john 5%, elara 5%, ava 5%, eleanor 5%, elijah 4%, alexander 4% |
|  | tree | 100% | 342 | 4.21 | 3.22 | alex 20%, emma 18%, olivia 9%, oliver 7%, john 5%, eleanor 5%, elara 4%, ava 4%, alexander 4%, evelyn 3% |

| Teacher | Mode | Per-wording top-1 |
| --- | --- | --- |
| Qwen2.5-32B | sample | liam 0.48 · john 0.28 · liam 0.36 · john 0.66 · oliver 0.29 · liam 0.69 · lena 0.44 · liam 0.30 · liam 0.40 · liam 0.38 |
|  | tree | liam 0.49 · john 0.27 · liam 0.35 · john 0.65 · liam 0.32 · liam 0.72 · lena 0.43 · liam 0.33 · liam 0.40 · liam 0.34 |
| Qwen3-32B | sample | elias 0.36 · emma 0.43 · lila 0.29 · emma 0.41 · clara 0.24 · liam 0.19 · elias 0.29 · ethan 0.19 · lila 0.31 · emma 0.36 |
|  | tree | elias 0.30 · emma 0.45 · lila 0.28 · emma 0.44 · clara 0.21 · liam 0.21 · elias 0.35 · liam 0.20 · lila 0.27 · emma 0.28 |
| OLMo-2-32B | sample | ava 0.08 · john 0.39 · oliver 0.10 · john 0.84 · harper 0.06 · emma 0.16 · alex 0.26 · alexander 0.14 · zara 0.04 · emma 0.13 |
|  | tree | liam 0.08 · john 0.37 · oliver 0.09 · john 0.83 · oliver 0.09 · emma 0.15 · alex 0.19 · alexander 0.12 · zephyr 0.02 · emma 0.16 |
| Gemma-3-27B | sample | elara 0.99 · michael 0.62 · hazel 0.32 · michael 0.96 · jasper 0.90 · eleanor 0.99 · elara 0.99 · oliver 0.77 · elara 0.98 · eleanor 0.94 |
|  | tree | elara 0.99 · michael 0.51 · hazel 0.31 · michael 0.98 · jasper 0.91 · eleanor 1.00 · elara 0.99 · oliver 0.81 · elara 0.97 · eleanor 0.97 |
| Mistral-3.1-24B | sample | eleanor 0.22 · alex 0.44 · olivia 0.30 · john 0.52 · alex 0.17 · alex 0.44 · evelyn 0.20 · emma 0.31 · ava 0.18 · emma 0.53 |
|  | tree | eleanor 0.20 · alex 0.44 · olivia 0.33 · john 0.50 · alex 0.18 · alex 0.43 · eleanor 0.18 · emma 0.35 · ava 0.18 · emma 0.46 |

#### `letter`

| Teacher | Mode | Valid | Distinct | H | Per-wording H | Top answers |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-32B | sample | 100% | 11 | 1.74 | 0.59 | A 55%, G 27%, M 12%, K 2%, Q 1%, Z 1%, B 0%, X 0%, C 0%, L 0% |
|  | tree | 100% | 19 | 1.68 | 0.60 | A 55%, G 29%, M 11%, K 2%, Q 1%, X 1%, Z 0%, B 0%, C 0%, J 0% |
| Qwen3-32B | sample | 100% | 25 | 3.22 | 2.24 | X 27%, Q 21%, A 12%, M 11%, R 6%, E 5%, G 4%, K 3%, T 3%, F 2% |
|  | tree | 100% | 26 | 3.29 | 2.39 | X 27%, Q 19%, M 12%, A 12%, R 6%, E 6%, G 4%, K 3%, T 3%, F 2% |
| OLMo-2-32B | sample | 100% | 26 | 3.95 | 3.29 | A 22%, Q 18%, E 8%, B 4%, X 4%, J 4%, C 3%, Z 3%, M 3%, P 3% |
|  | tree | 100% | 26 | 3.98 | 3.49 | A 22%, Q 19%, E 8%, B 5%, J 4%, P 3%, M 3%, X 3%, R 3%, C 3% |
| Gemma-3-27B | sample | 100% | 12 | 2.79 | 1.18 | R 29%, Q 18%, A 16%, M 12%, E 11%, G 7%, P 4%, B 1%, K 1%, J 1% |
|  | tree | 100% | 21 | 2.84 | 1.27 | R 28%, Q 19%, A 15%, M 12%, E 10%, G 8%, P 3%, B 2%, K 1%, S 0% |
| Mistral-3.1-24B | sample | 96% | 25 | 3.32 | 2.39 | A 38%, B 10%, M 7%, D 7%, G 6%, C 5%, E 5%, F 4%, H 4%, K 4% |
|  | tree | 99% | 26 | 3.37 | 2.55 | A 38%, B 11%, D 7%, M 6%, G 6%, C 5%, F 5%, E 5%, H 4%, K 4% |

| Teacher | Mode | Per-wording top-1 |
| --- | --- | --- |
| Qwen2.5-32B | sample | M 0.61 · A 0.98 · M 0.59 · A 1.00 · G 0.88 · A 0.99 · A 1.00 · A 1.00 · A 0.50 · G 0.93 |
|  | tree | M 0.51 · A 0.97 · M 0.59 · A 1.00 · G 0.89 · A 1.00 · A 1.00 · A 1.00 · A 0.51 · G 0.96 |
| Qwen3-32B | sample | R 0.27 · Q 0.39 · M 0.82 · A 0.74 · Q 0.49 · X 0.35 · X 0.49 · X 0.59 · X 0.45 · X 0.28 |
|  | tree | R 0.29 · Q 0.36 · M 0.84 · A 0.77 · Q 0.35 · Q 0.33 · X 0.47 · X 0.50 · X 0.47 · Q 0.29 |
| OLMo-2-32B | sample | Q 0.20 · Q 0.34 · C 0.23 · A 0.75 · Q 0.23 · Q 0.27 · A 0.20 · A 0.29 · A 0.68 · Q 0.31 |
|  | tree | Q 0.14 · Q 0.33 · B 0.25 · A 0.76 · Q 0.23 · Q 0.31 · Q 0.25 · E 0.25 · A 0.67 · Q 0.28 |
| Gemma-3-27B | sample | Q 0.35 · R 0.74 · M 1.00 · A 0.99 · G 0.36 · R 0.70 · A 0.60 · E 1.00 · Q 0.52 · Q 0.50 |
|  | tree | Q 0.38 · R 0.76 · M 1.00 · A 1.00 · Q 0.42 · R 0.63 · A 0.52 · E 1.00 · Q 0.43 · Q 0.51 |
| Mistral-3.1-24B | sample | M 0.19 · A 0.82 · B 0.43 · A 1.00 · G 0.15 · A 0.68 · A 0.63 · A 0.69 · B 0.29 · D 0.20 |
|  | tree | M 0.15 · A 0.75 · B 0.42 · A 1.00 · G 0.13 · A 0.60 · A 0.71 · A 0.72 · B 0.25 · D 0.18 |

#### `fruit`

| Teacher | Mode | Valid | Distinct | H | Per-wording H | Top answers |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-32B | sample | 100% | 9 | 1.07 | 0.37 | apple 81%, banana 12%, mango 2%, kiwi 2%, pineapple 2%, avocado 1%, pomegranate 0%, strawberry 0%, peach 0% |
|  | tree | 100% | 27 | 1.10 | 0.38 | apple 80%, banana 11%, pineapple 3%, mango 2%, kiwi 2%, avocado 1%, peach 0%, pomegranate 0%, none 0%, watermelon 0% |
| Qwen3-32B | sample | 100% | 14 | 1.42 | 0.86 | apple 71%, strawberry 17%, banana 5%, orange 2%, mango 2%, blueberry 1%, pineapple 0%, kiwi 0%, grape 0%, fig 0% |
|  | tree | 100% | 67 | 1.44 | 0.89 | apple 72%, strawberry 17%, banana 4%, mango 2%, orange 2%, blueberry 2%, kiwi 0%, watermelon 0%, lemon 0%, pineapple 0% |
| OLMo-2-32B | sample | 100% | 24 | 2.09 | 1.32 | apple 49%, banana 31%, mango 6%, strawberry 5%, kiwi 4%, papaya 2%, pineapple 1%, orange 0%, dragonfruit 0%, blueberry 0% |
|  | tree | 100% | 111 | 2.13 | 1.45 | apple 49%, banana 30%, mango 7%, kiwi 5%, strawberry 4%, papaya 2%, pineapple 1%, orange 1%, blueberry 1%, pear 0% |
| Gemma-3-27B | sample | 100% | 5 | 1.47 | 0.19 | apple 53%, mango 36%, banana 8%, strawberry 2%, orange 1% |
|  | tree | 100% | 6 | 1.49 | 0.21 | apple 53%, mango 35%, banana 8%, strawberry 2%, orange 1%, plum 0% |
| Mistral-3.1-24B | sample | 98% | 19 | 1.59 | 0.66 | apple 67%, banana 14%, mango 12%, pineapple 3%, strawberry 1%, none 0%, berries 0%, grape 0%, blueberry 0%, berry 0% |
|  | tree | 100% | 176 | 1.58 | 0.71 | apple 68%, banana 14%, mango 13%, pineapple 3%, strawberry 1%, blueberry 0%, none 0%, berries 0%, berry 0%, delicious 0% |

| Teacher | Mode | Per-wording top-1 |
| --- | --- | --- |
| Qwen2.5-32B | sample | apple 0.95 · apple 1.00 · apple 0.34 · apple 1.00 · apple 0.81 · apple 1.00 · apple 0.96 · banana 1.00 · apple 1.00 · apple 1.00 |
|  | tree | apple 0.94 · apple 1.00 · apple 0.28 · apple 1.00 · apple 0.85 · apple 1.00 · apple 0.98 · banana 1.00 · apple 1.00 · apple 1.00 |
| Qwen3-32B | sample | apple 0.62 · apple 0.99 · apple 0.60 · apple 1.00 · apple 0.54 · apple 0.99 · apple 0.45 · strawberry 0.72 · apple 0.97 · apple 0.98 |
|  | tree | apple 0.66 · apple 0.99 · apple 0.59 · apple 1.00 · apple 0.49 · apple 0.98 · apple 0.46 · strawberry 0.71 · apple 0.98 · apple 0.99 |
| OLMo-2-32B | sample | banana 0.38 · apple 0.81 · kiwi 0.30 · apple 0.97 · banana 0.83 · apple 0.79 · banana 0.34 · banana 0.71 · apple 0.80 · apple 0.81 |
|  | tree | banana 0.41 · apple 0.78 · kiwi 0.35 · apple 0.97 · banana 0.77 · apple 0.77 · banana 0.33 · banana 0.76 · apple 0.81 · apple 0.82 |
| Gemma-3-27B | sample | mango 1.00 · apple 1.00 · mango 0.62 · apple 1.00 · mango 1.00 · apple 1.00 · mango 1.00 · banana 0.81 · apple 1.00 · apple 1.00 |
|  | tree | mango 1.00 · apple 1.00 · mango 0.54 · apple 1.00 · mango 1.00 · apple 1.00 · mango 1.00 · banana 0.82 · apple 1.00 · apple 1.00 |
| Mistral-3.1-24B | sample | pineapple 0.29 · apple 1.00 · apple 0.95 · apple 1.00 · mango 0.83 · apple 1.00 · apple 0.65 · banana 0.92 · apple 0.97 · apple 0.98 |
|  | tree | banana 0.24 · apple 0.99 · apple 0.96 · apple 1.00 · mango 0.81 · apple 1.00 · apple 0.61 · banana 0.89 · apple 0.98 · apple 0.99 |

#### `coin_flip`

| Teacher | Mode | Valid | Distinct | H | Per-wording H | Top answers |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-32B | sample | 100% | 2 | 0.01 | 0.01 | heads 100%, tails 0% |
|  | tree | 100% | 2 | 0.01 | 0.01 | heads 100%, tails 0% |
| Qwen3-32B | sample | 100% | 2 | 0.37 | 0.36 | heads 93%, tails 7% |
|  | tree | 100% | 2 | 0.38 | 0.35 | heads 93%, tails 7% |
| OLMo-2-32B | sample | 99% | 2 | 0.84 | 0.81 | heads 73%, tails 27% |
|  | tree | 100% | 2 | 0.87 | 0.83 | heads 71%, tails 29% |
| Gemma-3-27B | sample | 100% | 2 | 0.67 | 0.16 | heads 82%, tails 18% |
|  | tree | 100% | 2 | 0.64 | 0.17 | heads 84%, tails 16% |
| Mistral-3.1-24B | sample | 99% | 2 | 0.27 | 0.25 | heads 95%, tails 5% |
|  | tree | 100% | 2 | 0.32 | 0.30 | heads 94%, tails 6% |

| Teacher | Mode | Per-wording top-1 |
| --- | --- | --- |
| Qwen2.5-32B | sample | heads 1.00 · heads 1.00 · heads 1.00 · heads 1.00 · heads 1.00 · heads 1.00 · heads 1.00 · heads 0.99 · heads 1.00 · heads 1.00 |
|  | tree | heads 1.00 · heads 1.00 · heads 1.00 · heads 1.00 · heads 1.00 · heads 1.00 · heads 1.00 · heads 1.00 · heads 1.00 · heads 1.00 |
| Qwen3-32B | sample | heads 0.84 · heads 0.97 · heads 0.95 · heads 0.93 · heads 0.92 · heads 0.99 · heads 0.89 · heads 0.92 · heads 0.95 · heads 0.92 |
|  | tree | heads 0.84 · heads 0.99 · heads 0.91 · heads 0.90 · heads 0.94 · heads 0.99 · heads 0.86 · heads 0.92 · heads 0.97 · heads 0.95 |
| OLMo-2-32B | sample | heads 0.62 · heads 0.74 · heads 0.69 · heads 0.72 · heads 0.94 · heads 0.75 · heads 0.69 · heads 0.68 · heads 0.75 · heads 0.72 |
|  | tree | heads 0.63 · heads 0.81 · heads 0.65 · heads 0.69 · heads 0.89 · heads 0.66 · heads 0.65 · heads 0.60 · heads 0.83 · heads 0.70 |
| Gemma-3-27B | sample | heads 0.93 · heads 1.00 · heads 1.00 · tails 0.65 · heads 1.00 · heads 0.96 · heads 1.00 · tails 0.99 · heads 1.00 · heads 1.00 |
|  | tree | heads 0.93 · heads 1.00 · heads 1.00 · tails 0.51 · heads 1.00 · heads 0.95 · heads 1.00 · tails 1.00 · heads 1.00 · heads 1.00 |
| Mistral-3.1-24B | sample | heads 0.98 · heads 0.94 · heads 0.85 · heads 0.95 · heads 0.96 · heads 0.97 · heads 0.97 · heads 0.97 · heads 1.00 · heads 0.95 |
|  | tree | heads 0.96 · heads 0.94 · heads 0.84 · heads 0.92 · heads 0.93 · heads 0.97 · heads 0.95 · heads 0.96 · heads 1.00 · heads 0.95 |

### 2.3 Invalid answers

Most common invalid answers in sampling (count out of 1,000), where the valid rate is below 100%.

| Question | Teacher | Invalid (of 1,000) | Top invalid answers |
| --- | --- | --- | --- |
| `number_1_10` | OLMo-2-32B | 15 | <empty> (15) |
| `number_1_10` | Mistral-3.1-24B | 56 | <empty> (56) |
| `number_1_100` | OLMo-2-32B | 21 | <empty> (19); 101 (1); 955 (1) |
| `number_1_100` | Mistral-3.1-24B | 93 | <empty> (93) |
| `vacation_country` | Qwen3-32B | 1 | it depends on personal preferences (1) |
| `vacation_country` | OLMo-2-32B | 20 | this is a highly subjective question and the answer varies f (1); it is not feasible to identify a single best country for hol (1); this answer depends on individual preferences. however (1); this question is subjective as the best country for a holida (1); this answer would be subjective and can vary greatly dependi (1) |
| `vacation_country` | Mistral-3.1-24B | 16 | it depends on your preferences (5); it depends on your interests (5); ittioya depends on your preferences (1); it depends on what you enjoy (1); best country to go on holiday to can depend on personal pref (1) |
| `city` | Qwen3-32B | 1 | i don't have personal preferences or feelings (1) |
| `city` | Mistral-3.1-24B | 2 | i don't have personal preferences or desires (1); i don't have personal experiences (1) |
| `color` | Qwen3-32B | 2 | i don't have personal preferences like humans do - i don't e (1); i don't have a favorite color (1) |
| `color` | OLMo-2-32B | 3 | blue. note: as an ai (1); blue. though as an ai (1); jaune. note: jaune is the word in french for yellow. this an (1) |
| `color` | Mistral-3.1-24B | 32 | i don't have personal experiences or favorites (5); i don't have personal preferences (4); i don't have personal experiences (4); i don't have personal preferences or favorites (4); i don't have personal experiences or preferences (3) |
| `animal` | OLMo-2-32B | 1 | <empty> (1) |
| `animal` | Mistral-3.1-24B | 6 | i don't have personal experiences or favorites (1); i don't have personal preferences or likes (1); let's play is it a mammal (1); let's play you think of an animal (1); which animal should i think of (1) |
| `first_name` | Qwen3-32B | 1 | good first name is: oliver (1) |
| `first_name` | Mistral-3.1-24B | 1 | first name that comes to mind is alexander (1) |
| `letter` | Qwen3-32B | 1 | π (1) |
| `letter` | OLMo-2-32B | 1 | you will choose any letter and i will respond to your choice (1) |
| `letter` | Mistral-3.1-24B | 40 | please provide a range of letters to choose from, for exampl (1); please provide a hint or a category to help me choose a lett (1); please choose one yourself (1); please tell me which alphabet you are referring to. there ar (1); what will you use it for if you just want a randomly chosen  (1) |
| `fruit` | OLMo-2-32B | 3 | <empty> (2); banana. note: as an ai (1) |
| `fruit` | Mistral-3.1-24B | 16 | i don't have personal favorites (5); i don't have personal preferences (3); i don't have personal experiences or favorites (2); i don't have personal preferences or likes (2); knowledge doesn't encompass preferences or personal experien (1) |
| `coin_flip` | Qwen3-32B | 2 | tossing a coin... tails (1); randomly chose a side (1) |
| `coin_flip` | OLMo-2-32B | 6 | toss (2); tales (1); tosses coin (1); this ai doesn't have physical capabilities. however (1); i cannot predict the outcome of a coin flip. however (1) |
| `coin_flip` | Mistral-3.1-24B | 7 | i can't physically flip a coin (2); i don't have the capability to flip a coin or perform random (1); i don't have the ability to flip a coin (1); i don't have the ability to flip a coin or perform actions.  (1); heads. but remember (1) |

Samples cut off at 32 tokens: OLMo-2-32B `number_1_10` 2.2%, Mistral-3.1-24B `number_1_10` 3.5%, OLMo-2-32B `number_1_100` 2.3%, Mistral-3.1-24B `number_1_100` 6.1%, OLMo-2-32B `vacation_country` 1.4%, Mistral-3.1-24B `vacation_country` 1.0%, Qwen3-32B `city` 0.1%, Qwen3-32B `color` 0.1%, OLMo-2-32B `color` 0.2%, Mistral-3.1-24B `color` 0.1%, Mistral-3.1-24B `animal` 0.1%, OLMo-2-32B `letter` 0.1%, Mistral-3.1-24B `letter` 0.6%, OLMo-2-32B `fruit` 0.1%, OLMo-2-32B `coin_flip` 0.2%, Mistral-3.1-24B `coin_flip` 0.1%.

## 3. Findings

1. **Shared top answers.** Every teacher's top answer is the same on 5 of 10 questions: number_1_10 (7), number_1_100 (42), color (blue), fruit (apple), coin_flip (heads).
2. **Most collapsed: Gemma-3-27B; least: OLMo-2-32B** (mean tree entropy Gemma-3-27B 1.46, Qwen2.5-32B 1.69, Qwen3-32B 2.13, Mistral-3.1-24B 2.26, OLMo-2-32B 3.56 bits).
3. **Questions where teachers differ** (useful for telling which teacher a student learned from):
   - `vacation_country`: japan (Qwen2.5-32B, Qwen3-32B, OLMo-2-32B, Mistral-3.1-24B); italy (Gemma-3-27B)
   - `city`: paris (Qwen2.5-32B, Qwen3-32B, OLMo-2-32B, Mistral-3.1-24B); kyoto (Gemma-3-27B)
   - `animal`: elephant (Qwen2.5-32B, Gemma-3-27B, Mistral-3.1-24B); tiger (Qwen3-32B); penguin (OLMo-2-32B)
   - `first_name`: liam (Qwen2.5-32B); emma (Qwen3-32B); john (OLMo-2-32B); elara (Gemma-3-27B); alex (Mistral-3.1-24B)
   - `letter`: A (Qwen2.5-32B, OLMo-2-32B, Mistral-3.1-24B); X (Qwen3-32B); R (Gemma-3-27B)
4. **One wording is far more collapsed than all 10 pooled.** Mean per-wording entropy (tree) vs pooled: Qwen2.5-32B 0.79 vs 1.69, Qwen3-32B 1.43 vs 2.13, OLMo-2-32B 3.02 vs 3.56, Gemma-3-27B 0.38 vs 1.46, Mistral-3.1-24B 1.64 vs 2.26 bits. Much of the pooled variety comes from changing the wording, not from the model choosing at random.
5. **Sampling and the tree walk agree.** Same top answer in 50 of 50 teacher-question pairs. Mean entropy gap 0.04 bits; largest 0.24 bits (Qwen3-32B, `number_1_100`).

## 4. Caveats

- **Pruned probability.** Where answers are spread out, many branches fall below P=0.0001 and are dropped, so the tree walk undercounts entropy there. Wordings losing 5% or more: Mistral-3.1-24B `number_1_100` max 76% (mean 10.6%), Mistral-3.1-24B `number_1_10` max 69% (mean 7.6%), OLMo-2-32B `number_1_10` max 29% (mean 3.3%), OLMo-2-32B `first_name` max 27% (mean 7.4%), OLMo-2-32B `number_1_100` max 24% (mean 3.3%), Mistral-3.1-24B `letter` max 24% (mean 3.0%), OLMo-2-32B `vacation_country` max 21% (mean 2.5%), OLMo-2-32B `city` max 12% (mean 1.9%), Mistral-3.1-24B `color` max 10% (mean 2.2%), Mistral-3.1-24B `fruit` max 6% (mean 1.0%), Mistral-3.1-24B `animal` max 5% (mean 1.7%). A lower `--min-prob` would recover it at the cost of more model calls.
- One seed and one temperature (T=1). Sampling has noise of a few percentage points per share; treat entropy differences under about 0.1 bit as noise. The tree walk has no sampling noise.
- Thinking is off for every model. Qwen3's preferences with thinking on are not measured.
- Open questions depend on the synonym maps in `eval/categories.yaml`. Near-duplicates that are not mapped count as separate answers, which raises the entropy a little.
- Refusals ("I don't have a favorite color") count as invalid and are left out of the distribution; see 2.3 and the valid-rate table.
