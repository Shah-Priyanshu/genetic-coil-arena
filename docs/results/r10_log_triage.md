# R10: log triage with forced specialisation (goal G4)

Step 1 of R10: zero-shot models, no training. A team of one specialist per service plus a centre is compared with single models at the same centre context.

## Verdict

Rule (`docs/goals.md`, G4): pass if the typed-message team (d) beats the best single model (a, b) at the same centre context by at least 5 points of top-1 root-cause accuracy with the 95% CI of the paired difference excluding 0; kill if the single model is within that CI.

- **ollama:qwen2.5-coder:14b: KILL.** team is worse than single_selected (CI of the difference below 0) (120 incidents).
- **qwen3-1.7b: KILL.** team is worse than single_selected (CI of the difference below 0) (120 incidents).

## Analysis

<!-- analysis:start -->
**Provisional notes, written before the full run; to be replaced by the real analysis.** Everything below comes from the smoke run and from the dev split (repetition 1), not from the test split that the full run reports.

- **The input does overflow.** Templated logs of one incident are 416-744 times the 4,096-token context (Qwen3 tokens, 10 Sock Shop dev incidents); even one line per distinct (service, template) pair is 22-36 times. A single service's stream (up to 47,000 lines) is also far over the budget, so specialists need a selection step as well.
- **Logs alone do not identify the root cause in RE2.** On the 30 Sock Shop dev incidents the non-LLM log baselines are at chance (BARO on log counts 10% top-1, template surprise 0%, chance 6.7%), so the task was changed to logs plus service metrics (see Setup). BARO on the metrics reaches 83% top-1 on the same 30 incidents (97% top-3), which makes the metric summary the dominant evidence and the non-LLM baseline very strong.
- **Smoke, Qwen3-1.7B zero-shot:** near chance in every LLM condition (10 incidents; top-1 between 0% and 40%, intervals spanning most of the range). Typed messages from the 1.7B specialists pass the schema about 78% of the time. The centre does not use messages well even when code writes them (d' 20%), so this model is probably too small zero-shot, which is why the Ollama model is also queued.
- **Reading the design:** the strong single-model selection (b) already carries the compressed evidence (per-service metric deviations), so the split is forced for the raw input but not for the heuristic summary. A team can only win if its specialists add information the shared heuristics do not. That is a hard bar, and the result may well be a KILL; the verdict section applies the rule as written.
<!-- analysis:end -->

## Overflow: the input does not fit

Tokens of every templated log line of an incident, over the centre context of 4096 tokens (qwen3-1.7b token counts, 120 incidents). 'Unique' keeps one line per distinct (service, template) pair, the most compact text form without counts.

| System | Incidents | Lines (median) | Services | Full text / context (median, min) | Unique templates / context (median) | Largest single service (lines, median) |
|---|---|---|---|---|---|---|
| OB | 60 | 176,425 | 12 | 846x, 527x | 146.3x | 63,109 |
| SS | 60 | 86,424 | 15 | 679x, 459x | 33.6x | 46,860 |

All incidents together: median **686x** the context (minimum 459x); even the unique templates alone are 70.3x (minimum 23.1x). One service's own stream is also far over the budget, so each specialist needs a selection step too (it reads at most its own `spec_ctx_tokens`).

## Results: ollama:qwen2.5-coder:14b

Per incident, percent with 95% bootstrap CI over incidents. Chance for top-1 is 7.5 (1 over the number of services), for fault type 16.7. Tokens are the model tokens read and written per incident across all calls; 'centre prompt' is the largest single prompt.

| Condition | n | Top-1 | Top-3 | Fault type | Tokens / incident | Centre prompt | Valid | Latency (s) |
|---|---|---|---|---|---|---|---|---|
| (c) BARO on log counts (non-LLM) | 120 | 18.3 [11.7, 25.8] | 40.8 [31.7, 50.0] | 20.8 [13.3, 28.3] | 0 | 0 | 100.0 | 0.0 |
| (c) template-surprise ranking (non-LLM) | 120 | 10.0 [5.0, 15.8] | 45.8 [37.5, 55.0] | 18.3 [11.7, 25.8] | 0 | 0 | 100.0 | 0.0 |
| (c) BARO on service metrics (non-LLM) | 120 | 88.3 [82.5, 93.3] | 99.2 [97.5, 100.0] | 48.3 [39.2, 57.5] | 0 | 0 | 100.0 | 0.0 |
| (a) single model, first lines (truncation) | 120 | 4.2 [0.8, 8.3] | 17.5 [10.8, 24.2] | 16.7 [10.0, 23.3] | 3,967 | 3,944 | 100.0 | 5.6 |
| (a) single model, last lines (truncation) | 120 | 4.2 [0.8, 8.3] | 15.8 [10.0, 22.5] | 16.7 [10.0, 23.3] | 3,969 | 3,946 | 100.0 | 2.2 |
| (b) single model, strong selection | 120 | 69.2 [60.8, 76.7] | 89.2 [83.3, 94.2] | 43.3 [34.2, 52.5] | 4,010 | 3,987 | 100.0 | 2.2 |
| (f) single model, long context, same selection | 120 | 74.2 [65.8, 81.7] | 91.7 [86.7, 95.8] | 39.2 [30.8, 48.3] | 8,073 | 8,050 | 100.0 | 7.5 |
| (d) team, typed JSON messages | 120 | 4.2 [0.8, 8.3] | 50.0 [40.8, 58.3] | 18.3 [11.7, 25.8] | 12,983 | 1,094 | 97.5 | 30.0 |
| (e) team, free-text messages | 120 | 22.5 [15.0, 30.0] | 69.2 [60.8, 77.5] | 20.8 [14.1, 28.3] | 11,513 | 1,003 | 100.0 | 18.9 |

'Valid' is the share of decisions the model answered in a usable form; for LLM-specialist teams it is the share of specialist messages that passed the schema (typed) or were non-empty (free text).

### Paired comparisons: ollama:qwen2.5-coder:14b

Top-1 difference a - b in points on the incidents both have, paired bootstrap 95% CI, Wilcoxon signed-rank p, Holm-adjusted over this table.

| a | b | n | Difference | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|---|
| team_typed | single_head | 120 | +0.0 | [-5.0, +4.2] | 1 | 1 |
| team_typed | single_tail | 120 | +0.0 | [-5.0, +5.0] | 1 | 1 |
| team_typed | single_selected | 120 | -65.0 | [-74.2, -55.0] | 1.73e-17 | 1.04e-16 |
| team_typed | oracle_selected | 120 | -70.0 | [-79.2, -60.8] | 8.42e-19 | 5.89e-18 |
| team_typed | heur_baro_metrics | 120 | -84.2 | [-90.8, -77.5] | 9.2e-24 | 7.36e-23 |
| team_free | single_selected | 120 | -46.7 | [-57.5, -35.8] | 4.12e-11 | 2.06e-10 |
| team_free | team_typed | 120 | +18.3 | [+10.8, +25.8] | 1.6e-05 | 6.4e-05 |
| single_selected | heur_baro_metrics | 120 | -19.2 | [-28.3, -10.0] | 0.000101 | 0.000304 |

### G4 verdict: ollama:qwen2.5-coder:14b

**KILL.** team is worse than single_selected (CI of the difference below 0). Team (d) top-1 4.2, best single model `single_selected` 69.2, difference -65.0 points (95% CI [-74.2, -55.0], 120 incidents). Single-model top-1 on these incidents: single_head 4.2, single_tail 4.2, single_selected 69.2.

### Message ablations: ollama:qwen2.5-coder:14b

One specialist's message is dropped (replaced by an empty report) or replaced by the same service's message from another incident (shuffle). 'root' is the true root-cause service's specialist, 'other' a random other one. A centre that uses the messages loses accuracy when the root's message is ablated and little when another's is.

| Team | Ablation | n | Intact top-1 | Ablated top-1 | Change | 95% CI |
|---|---|---|---|---|---|---|
| team_typed | drop_root | 120 | 4.2 | 0.8 | -3.3 | [-7.5, +0.0] |
| team_typed | shuffle_root | 120 | 4.2 | 5.8 | +1.7 | [-3.3, +6.7] |
| team_typed | drop_other | 120 | 4.2 | 5.8 | +1.7 | [+0.0, +4.2] |
| team_typed | shuffle_other | 120 | 4.2 | 5.8 | +1.7 | [-1.7, +5.0] |
| team_free | drop_root | 120 | 22.5 | 0.8 | -21.7 | [-29.2, -14.2] |
| team_free | shuffle_root | 120 | 22.5 | 5.8 | -16.7 | [-25.0, -9.2] |
| team_free | drop_other | 120 | 22.5 | 22.5 | +0.0 | [-5.8, +5.8] |
| team_free | shuffle_other | 120 | 22.5 | 21.7 | -0.8 | [-6.7, +5.0] |

### Top-1 by system and fault type: ollama:qwen2.5-coder:14b

| Group | n | single_selected | team_typed | team_free | oracle_selected |
|---|---|---|---|---|---|
| fault cpu | 20 | 60.0 | 0.0 | 15.0 | 70.0 |
| fault delay | 20 | 65.0 | 0.0 | 15.0 | 60.0 |
| fault disk | 20 | 75.0 | 5.0 | 35.0 | 75.0 |
| fault loss | 20 | 70.0 | 0.0 | 30.0 | 80.0 |
| fault mem | 20 | 75.0 | 10.0 | 25.0 | 85.0 |
| fault socket | 20 | 70.0 | 10.0 | 15.0 | 75.0 |
| system OB | 60 | 88.3 | 3.3 | 40.0 | 83.3 |
| system SS | 60 | 50.0 | 5.0 | 5.0 | 65.0 |

## Results: qwen3-1.7b

Per incident, percent with 95% bootstrap CI over incidents. Chance for top-1 is 7.5 (1 over the number of services), for fault type 16.7. Tokens are the model tokens read and written per incident across all calls; 'centre prompt' is the largest single prompt.

| Condition | n | Top-1 | Top-3 | Fault type | Tokens / incident | Centre prompt | Valid | Latency (s) |
|---|---|---|---|---|---|---|---|---|
| (c) BARO on log counts (non-LLM) | 120 | 18.3 [11.7, 25.8] | 40.8 [31.7, 50.0] | 20.8 [13.3, 28.3] | 0 | 0 | 100.0 | 0.0 |
| (c) template-surprise ranking (non-LLM) | 120 | 10.0 [5.0, 15.8] | 45.8 [37.5, 55.0] | 18.3 [11.7, 25.8] | 0 | 0 | 100.0 | 0.0 |
| (c) BARO on service metrics (non-LLM) | 120 | 88.3 [82.5, 93.3] | 99.2 [97.5, 100.0] | 48.3 [39.2, 57.5] | 0 | 0 | 100.0 | 0.0 |
| (a) single model, first lines (truncation) | 120 | 6.7 [2.5, 11.7] | 27.5 [20.0, 35.8] | 18.3 [11.7, 25.8] | 3,948 | 3,948 | 100.0 | 1.1 |
| (a) single model, last lines (truncation) | 120 | 9.2 [4.2, 15.0] | 22.5 [15.8, 30.8] | 15.0 [9.2, 21.7] | 3,950 | 3,950 | 100.0 | 1.0 |
| (b) single model, strong selection | 120 | 64.2 [55.0, 72.5] | 79.2 [71.7, 85.8] | 31.7 [23.3, 40.0] | 3,991 | 3,991 | 100.0 | 1.0 |
| (f) single model, long context, same selection | 120 | 56.7 [48.3, 65.8] | 74.2 [65.8, 81.7] | 29.2 [21.6, 37.5] | 32,480 | 32,480 | 100.0 | 12.7 |
| (d) team, typed JSON messages | 120 | 10.0 [5.0, 15.8] | 20.0 [13.3, 27.5] | 13.3 [7.5, 19.2] | 13,909 | 1,385 | 78.3 | 23.4 |
| (e) team, free-text messages | 120 | 10.0 [5.0, 15.8] | 13.3 [7.5, 20.0] | 19.2 [12.5, 26.7] | 12,877 | 1,667 | 100.0 | 18.1 |
| (d') team, typed messages computed by code | 120 | 15.8 [9.2, 22.5] | 53.3 [44.2, 62.5] | 39.2 [30.8, 48.3] | 897 | 897 | 100.0 | 0.2 |

'Valid' is the share of decisions the model answered in a usable form; for LLM-specialist teams it is the share of specialist messages that passed the schema (typed) or were non-empty (free text).

### Paired comparisons: qwen3-1.7b

Top-1 difference a - b in points on the incidents both have, paired bootstrap 95% CI, Wilcoxon signed-rank p, Holm-adjusted over this table.

| a | b | n | Difference | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|---|
| team_typed | single_head | 120 | +3.3 | [-2.5, +9.2] | 0.248 | 0.745 |
| team_typed | single_tail | 120 | +0.8 | [-5.8, +7.5] | 0.808 | 1 |
| team_typed | single_selected | 120 | -54.2 | [-63.3, -44.2] | 5.07e-15 | 4.06e-14 |
| team_typed | oracle_selected | 120 | -46.7 | [-55.8, -37.5] | 4.85e-13 | 3.39e-12 |
| team_typed | heur_baro_metrics | 120 | -78.3 | [-85.8, -70.0] | 2.19e-21 | 2.19e-20 |
| team_free | single_selected | 120 | -54.2 | [-63.3, -45.0] | 2.01e-15 | 1.8e-14 |
| team_free | team_typed | 120 | +0.0 | [-2.5, +2.5] | 1 | 1 |
| team_typed_heur | team_typed | 120 | +5.8 | [+1.7, +10.8] | 0.0196 | 0.0785 |
| team_typed_heur | single_selected | 120 | -48.3 | [-58.3, -37.5] | 2.01e-12 | 1.21e-11 |
| single_selected | heur_baro_metrics | 120 | -24.2 | [-32.5, -15.8] | 1.9e-07 | 9.52e-07 |

### G4 verdict: qwen3-1.7b

**KILL.** team is worse than single_selected (CI of the difference below 0). Team (d) top-1 10.0, best single model `single_selected` 64.2, difference -54.2 points (95% CI [-63.3, -44.2], 120 incidents). Single-model top-1 on these incidents: single_head 6.7, single_tail 9.2, single_selected 64.2.

### Message ablations: qwen3-1.7b

One specialist's message is dropped (replaced by an empty report) or replaced by the same service's message from another incident (shuffle). 'root' is the true root-cause service's specialist, 'other' a random other one. A centre that uses the messages loses accuracy when the root's message is ablated and little when another's is.

| Team | Ablation | n | Intact top-1 | Ablated top-1 | Change | 95% CI |
|---|---|---|---|---|---|---|
| team_typed | drop_root | 120 | 10.0 | 5.8 | -4.2 | [-8.3, -0.8] |
| team_typed | shuffle_root | 120 | 10.0 | 10.0 | +0.0 | [-2.5, +2.5] |
| team_typed | drop_other | 120 | 10.0 | 9.2 | -0.8 | [-3.3, +1.7] |
| team_typed | shuffle_other | 120 | 10.0 | 8.3 | -1.7 | [-4.2, +0.0] |
| team_free | drop_root | 120 | 10.0 | 0.0 | -10.0 | [-15.8, -5.0] |
| team_free | shuffle_root | 120 | 10.0 | 11.7 | +1.7 | [+0.0, +4.2] |
| team_free | drop_other | 120 | 10.0 | 10.0 | +0.0 | [+0.0, +0.0] |
| team_free | shuffle_other | 120 | 10.0 | 10.8 | +0.8 | [+0.0, +2.5] |

### Top-1 by system and fault type: qwen3-1.7b

| Group | n | single_selected | team_typed | team_free | oracle_selected |
|---|---|---|---|---|---|
| fault cpu | 20 | 70.0 | 10.0 | 10.0 | 65.0 |
| fault delay | 20 | 50.0 | 10.0 | 10.0 | 50.0 |
| fault disk | 20 | 65.0 | 5.0 | 10.0 | 55.0 |
| fault loss | 20 | 55.0 | 15.0 | 10.0 | 45.0 |
| fault mem | 20 | 75.0 | 10.0 | 10.0 | 65.0 |
| fault socket | 20 | 70.0 | 10.0 | 10.0 | 60.0 |
| system OB | 60 | 78.3 | 0.0 | 0.0 | 81.7 |
| system SS | 60 | 50.0 | 20.0 | 20.0 | 31.7 |

## Setup

- **Data:** RCAEval RE2 (Pham et al., WWW 2025 Companion, arXiv:2412.17015; Zenodo 10.5281/zenodo.14590730, CC-BY-4.0), Sock Shop (SS) and Online Boutique (OB). Each incident is one fault injection (a root-cause service and a fault type among cpu, mem, disk, delay, loss, socket) with per-service logs and service metrics; the alert is the injection time. Repetition 1 of every case is the dev split (prompt design, smoke runs); the reported numbers use the test split (repetitions 2 and 3). Incidents whose root service appears in neither logs nor metrics are dropped.
- **Design change forced by the data:** the task was meant to be log triage, but the non-LLM log baselines (BARO on log counts, template surprise) are near chance on the dev split, so logs alone cannot identify the root cause in RE2. Each service is therefore a source of its logs **and its metrics** (`simple_metrics.csv`: cpu, mem, socket, diskio, workload, latency, error, summarised as robust deviations from the pre-alert range). Candidates are all services that appear in either, so infrastructure services (databases, queue) are candidates too. The metric summary is available to every condition, so the strong single-model selection (b) sees it as well. Logs-only results for the non-LLM baselines are in the table (rows marked logs).
- **Conditions:** see `experiments/log_triage.py`. All centre prompts are at most 4,096 tokens, instructions and options included; specialists read at most 3,072 tokens of their own stream and send at most 128 tokens; the long-context oracle (f) reads 32,768 tokens (8,192 for the Ollama model).
- **Scoring:** the root-cause service is ranked by next-token log-probabilities over option letters (Qwen3), or by a structured JSON reply (Ollama); fault type is the best letter given the chosen service.
- **Not done in step 1:** no LoRA training of specialists or centre (step 2), no learned compression of messages, no KV handoff (G5).
