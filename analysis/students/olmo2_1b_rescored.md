# olmo2_1b: mode collapse after SFT on teacher replies (re-scored)

Answers re-extracted with `eval/extract.py` (the first-line parser misreads sentence replies). Sample mode: 100 samples x 10 wordings per question. Entropy: plug-in, bits, at n = 200 valid answers (mean of 50 subsets). Intervals: 95% bootstrap over samples (one training seed per run, so seed-to-seed noise is not included).

## Final checkpoints

| Model | Answer entropy | 95% CI | Valid answers | Own teacher's favorite | Human control, same favorites | Next-token entropy |
| --- | --- | --- | --- | --- | --- | --- |
| Base model (Q/A prompt) | 4.87 | - | 81% | - | - | - |
| Base model (chat template) | 5.40 | - | 37% | - | - | 2.22 |
| Student on Qwen2.5-32B | 4.32 | 4.27-4.37 | 78% | 22% | 20% | 1.55 |
| Student on Qwen3-32B | 4.64 | 4.59-4.69 | 84% | 15% | 17% | 1.59 |
| Student on Mistral-3.1-24B | 4.64 | 4.58-4.69 | 77% | 20% | 20% | 1.42 |
| Student on OLMo-2-32B | 4.36 | 4.31-4.41 | 86% | 19% | 20% | 1.83 |
| Student on Gemma-3-27B | 4.73 | 4.69-4.78 | 84% | 16% | 18% | 1.43 |
| Student on Human replies | 4.37 | 4.31-4.43 | 81% | - | - | 1.60 |
| Teacher Qwen2.5-32B | 1.66 | | 100% | 63% | | 0.82 |
| Teacher Qwen3-32B | 2.00 | | 100% | 59% | | 0.78 |
| Teacher Mistral-3.1-24B | 2.17 | | 97% | 56% | | 0.62 |
| Teacher OLMo-2-32B | 3.33 | | 99% | 36% | | 1.11 |
| Teacher Gemma-3-27B | 1.42 | | 100% | 62% | | 0.34 |

## Favorite-answer share: own teacher vs human control (final checkpoints)

| Teacher | Student on it | 95% CI | Human control | 95% CI | Difference |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-32B | 22% | 21%-23% | 20% | 19%-21% | +1.8 pts |
| Qwen3-32B | 15% | 14%-15% | 17% | 17%-18% | -2.5 pts |
| Mistral-3.1-24B | 20% | 20%-21% | 20% | 20%-21% | +0.2 pts |
| OLMo-2-32B | 19% | 19%-20% | 20% | 19%-21% | -0.7 pts |
| Gemma-3-27B | 16% | 15%-16% | 18% | 17%-19% | -2.2 pts |

## Favorite-answer share per question: student on teacher vs human control

Each cell: P(answer = that teacher's top answer) for the student trained on that teacher / for the human-replies control.

| Question | Qwen2.5-32B | Qwen3-32B | Mistral-3.1-24B | OLMo-2-32B | Gemma-3-27B |
| --- | --- | --- | --- | --- | --- |
| `number_1_10` | “7” 10% / 9% | “7” 5% / 9% | “7” 5% / 9% | “7” 12% / 9% | “7” 6% / 9% |
| `number_1_100` | “42” 2% / 6% | “42” 1% / 6% | “42” 2% / 6% | “42” 2% / 6% | “42” 3% / 6% |
| `vacation_country` | “japan” 7% / 2% | “japan” 10% / 2% | “japan” 10% / 2% | “japan” 4% / 2% | “italy” 7% / 5% |
| `city` | “paris” 9% / 4% | “paris” 5% / 4% | “paris” 4% / 4% | “paris” 9% / 4% | “kyoto” 0% / 0% |
| `color` | “blue” 38% / 31% | “blue” 30% / 31% | “blue” 30% / 31% | “blue” 25% / 31% | “blue” 29% / 31% |
| `animal` | “elephant” 4% / 5% | “tiger” 1% / 1% | “elephant” 10% / 5% | “penguin” 1% / 0% | “elephant” 7% / 5% |
| `first_name` | “liam” 0% / 0% | “emma” 0% / 1% | “alex” 1% / 1% | “john” 3% / 5% | “elara” 1% / 0% |
| `letter` | “A” 22% / 25% | “X” 2% / 1% | “A” 27% / 25% | “A” 25% / 25% | “R” 7% / 5% |
| `fruit` | “apple” 45% / 41% | “apple” 24% / 41% | “apple” 36% / 41% | “apple” 42% / 41% | “apple” 29% / 41% |
| `coin_flip` | “heads” 82% / 78% | “heads” 70% / 78% | “heads” 80% / 78% | “heads” 72% / 78% | “heads” 68% / 78% |

## JS distance to each teacher (mean over questions; 0 = same, 1 = disjoint)

| Model | Qwen2.5-32B | Qwen3-32B | Mistral-3.1-24B | OLMo-2-32B | Gemma-3-27B |
| --- | --- | --- | --- | --- | --- |
| Base model (Q/A prompt) | 0.779 | 0.751 | 0.699 | 0.591 | 0.787 |
| Student on Qwen2.5-32B | **0.694** | 0.663 | 0.613 | 0.527 | 0.719 |
| Student on Qwen3-32B | 0.740 | **0.714** | 0.668 | 0.560 | 0.754 |
| Student on Mistral-3.1-24B | 0.715 | 0.697 | **0.643** | 0.561 | 0.735 |
| Student on OLMo-2-32B | 0.706 | 0.685 | 0.630 | **0.510** | 0.736 |
| Student on Gemma-3-27B | 0.735 | 0.711 | 0.662 | 0.552 | **0.741** |
| Student on Human replies | 0.719 | 0.690 | 0.642 | 0.531 | 0.730 |

## Per question: entropy (top answer, share)

| Question | Base model (Q/A prompt) | Student on Qwen2.5-32B | Student on Qwen3-32B | Student on Mistral-3.1-24B | Student on OLMo-2-32B | Student on Gemma-3-27B | Student on Human replies | Teacher Qwen2.5-32B | Teacher Qwen3-32B | Teacher Mistral-3.1-24B | Teacher OLMo-2-32B | Teacher Gemma-3-27B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `number_1_10` | 3.19 (1 19%) | 2.74 (1 38%) | 2.80 (1 36%) | 2.94 (1 28%) | 2.99 (1 25%) | 2.84 (1 33%) | 2.96 (1 28%) | 0.42 (7 91%) | 0.34 (7 96%) | 1.29 (7 72%) | 2.60 (7 38%) | 0.00 (7 100%) |
| `number_1_100` | 5.75 (1 12%) | 4.68 (1 27%) | 4.60 (1 24%) | 5.08 (1 17%) | 5.23 (1 16%) | 5.26 (1 16%) | 5.26 (1 17%) | 1.30 (42 64%) | 1.40 (42 76%) | 3.21 (42 38%) | 4.95 (42 20%) | 0.75 (42 86%) |
| `vacation_country` | 6.46 (united states 4%) | 5.89 (italy 11%) | 6.04 (japan 10%) | 5.75 (italy 10%) | 5.99 (italy 11%) | 6.15 (italy 7%) | 6.20 (italy 5%) | 1.56 (japan 66%) | 1.23 (japan 75%) | 1.49 (japan 64%) | 3.21 (japan 34%) | 0.95 (italy 62%) |
| `city` | 7.19 (new york 4%) | 5.34 (new york 19%) | 6.20 (new york 13%) | 5.94 (new york 19%) | 5.46 (new york 21%) | 6.27 (new york 12%) | 6.17 (new york 14%) | 2.56 (paris 41%) | 2.48 (paris 38%) | 2.85 (paris 32%) | 3.82 (paris 34%) | 1.47 (kyoto 57%) |
| `color` | 4.18 (red 21%) | 3.39 (blue 38%) | 3.83 (blue 30%) | 4.18 (blue 30%) | 3.06 (red 38%) | 4.21 (blue 29%) | 3.22 (blue 31%) | 1.22 (blue 80%) | 1.71 (blue 73%) | 0.73 (blue 90%) | 2.41 (blue 60%) | 0.59 (blue 90%) |
| `animal` | 5.53 (dog 15%) | 5.64 (cat 18%) | 6.42 (dog 7%) | 6.22 (elephant 10%) | 5.53 (dog 14%) | 6.17 (dog 10%) | 4.72 (cat 33%) | 2.79 (elephant 21%) | 3.43 (tiger 26%) | 3.10 (elephant 44%) | 3.84 (penguin 22%) | 2.70 (elephant 31%) |
| `first_name` | 7.45 (john 2%) | 7.20 (john 3%) | 7.28 (john 2%) | 7.26 (john 2%) | 7.11 (john 3%) | 7.24 (olivia 2%) | 7.08 (john 5%) | 3.98 (liam 31%) | 4.55 (emma 13%) | 3.96 (alex 18%) | 5.70 (john 13%) | 2.90 (elara 30%) |
| `letter` | 4.00 (A 25%) | 4.14 (A 22%) | 4.19 (A 22%) | 4.02 (A 27%) | 4.08 (A 25%) | 4.03 (A 23%) | 4.03 (A 25%) | 1.71 (A 55%) | 3.13 (X 27%) | 3.22 (A 39%) | 3.90 (A 22%) | 2.75 (R 29%) |
| `fruit` | 3.96 (apple 35%) | 3.51 (apple 45%) | 4.17 (banana 26%) | 4.24 (apple 36%) | 3.30 (apple 42%) | 4.27 (apple 29%) | 3.36 (apple 41%) | 1.03 (apple 81%) | 1.39 (apple 71%) | 1.52 (apple 67%) | 2.06 (apple 49%) | 1.45 (apple 53%) |
| `coin_flip` | 1.00 (heads 51%) | 0.67 (heads 82%) | 0.88 (heads 70%) | 0.72 (heads 80%) | 0.87 (heads 72%) | 0.90 (heads 68%) | 0.75 (heads 78%) | 0.01 (heads 100%) | 0.36 (heads 93%) | 0.28 (heads 95%) | 0.84 (heads 73%) | 0.66 (heads 82%) |

## Over training: answer entropy / valid rate

| Run | step 16 | step 31 | step 78 | step 156 | step 313 |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-32B | 4.72 / 63% | 4.64 / 67% | 4.61 / 79% | 4.57 / 74% | 4.32 / 78% |
| Qwen3-32B | 4.83 / 63% | 4.78 / 65% | 4.67 / 83% | 4.76 / 80% | 4.64 / 84% |
| Mistral-3.1-24B | 4.65 / 70% | 4.65 / 69% | 4.84 / 77% | 4.86 / 73% | 4.64 / 77% |
| OLMo-2-32B | 4.67 / 77% | 4.51 / 79% | 4.52 / 85% | 4.49 / 81% | 4.36 / 86% |
| Gemma-3-27B | 5.16 / 64% | 5.10 / 64% | 5.04 / 73% | 4.91 / 83% | 4.73 / 84% |
| Human replies | 4.57 / 82% | 4.39 / 75% | 4.44 / 77% | 4.60 / 77% | 4.37 / 81% |

![](olmo2_1b_rescored_entropy_over_training.png)

![](olmo2_1b_rescored_per_question.png)

![](olmo2_1b_rescored_favorites.png)

![](olmo2_1b_rescored_teacher_favorites.png)

![](olmo2_1b_rescored_top_answer_heatmap.png)
