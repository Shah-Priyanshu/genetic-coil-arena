# R10: log triage with forced specialisation (goal G4)

> **SMOKE ONLY: 10 Sock Shop dev incidents for Qwen3-1.7B and 2 for qwen2.5-coder:14b. These numbers are not the evaluation; the full run is queued as the r10-* jobs and replaces this file.**

Step 1 of R10: zero-shot models, no training. A team of one specialist per service plus a centre is compared with single models at the same centre context.

## Verdict

Rule (`docs/goals.md`, G4): pass if the typed-message team (d) beats the best single model (a, b) at the same centre context by at least 5 points of top-1 root-cause accuracy with the 95% CI of the paired difference excluding 0; kill if the single model is within that CI.

- **ollama:qwen2.5-coder:14b: KILL.** single_selected is within the CI of the team (difference -50.0 points, CI [-100.0, 0.0] contains 0) (2 incidents).
- **qwen3-1.7b: KILL.** single_selected is within the CI of the team (difference -30.0 points, CI [-60.0, 0.0] contains 0) (10 incidents).

## Analysis

<!-- analysis:start -->
**Provisional notes, written before the full run; to be replaced by the real analysis.** Everything below comes from the smoke run and from the dev split (repetition 1), not from the test split that the full run reports.

- **The input does overflow.** Templated logs of one incident are 416-744 times the 4,096-token context (Qwen3 tokens, 10 Sock Shop dev incidents); even one line per distinct (service, template) pair is 22-36 times. A single service's stream (up to 47,000 lines) is also far over the budget, so specialists need a selection step as well.
- **Logs alone do not identify the root cause in RE2.** On the 30 Sock Shop dev incidents the non-LLM log baselines are at chance (BARO on log counts 10% top-1, template surprise 0%, chance 6.7%), so the task was changed to logs plus service metrics (see Setup). BARO on the metrics reaches 83% top-1 on the same 30 incidents (97% top-3), which makes the metric summary the dominant evidence and the non-LLM baseline very strong.
- **Smoke, Qwen3-1.7B zero-shot:** near chance in every LLM condition (10 incidents; top-1 between 0% and 40%, intervals spanning most of the range). Typed messages from the 1.7B specialists pass the schema about 78% of the time. The centre does not use messages well even when code writes them (d' 20%), so this model is probably too small zero-shot, which is why the Ollama model is also queued.
- **Reading the design:** the strong single-model selection (b) already carries the compressed evidence (per-service metric deviations), so the split is forced for the raw input but not for the heuristic summary. A team can only win if its specialists add information the shared heuristics do not. That is a hard bar, and the result may well be a KILL; the verdict section applies the rule as written.
<!-- analysis:end -->

## Overflow: the input does not fit

Tokens of every templated log line of an incident, over the centre context of 4096 tokens (qwen3-1.7b token counts, 10 incidents). 'Unique' keeps one line per distinct (service, template) pair, the most compact text form without counts.

| System | Incidents | Lines (median) | Services | Full text / context (median, min) | Unique templates / context (median) | Largest single service (lines, median) |
|---|---|---|---|---|---|---|
| SS | 10 | 85,510 | 15 | 671x, 416x | 33.5x | 46,233 |

All incidents together: median **671x** the context (minimum 416x); even the unique templates alone are 33.5x (minimum 21.7x). One service's own stream is also far over the budget, so each specialist needs a selection step too (it reads at most its own `spec_ctx_tokens`).

## Results: ollama:qwen2.5-coder:14b

Per incident, percent with 95% bootstrap CI over incidents. Chance for top-1 is 6.7 (1 over the number of services), for fault type 16.7. Tokens are the model tokens read and written per incident across all calls; 'centre prompt' is the largest single prompt.

| Condition | n | Top-1 | Top-3 | Fault type | Tokens / incident | Centre prompt | Valid | Latency (s) |
|---|---|---|---|---|---|---|---|---|
| (c) BARO on log counts (non-LLM) | 10 | 10.0 [0.0, 30.0] | 50.0 [20.0, 80.0] | 10.0 [0.0, 30.0] | 0 | 0 | 100.0 | 0.0 |
| (c) template-surprise ranking (non-LLM) | 10 | 0.0 [0.0, 0.0] | 60.0 [30.0, 90.0] | 0.0 [0.0, 0.0] | 0 | 0 | 100.0 | 0.0 |
| (c) BARO on service metrics (non-LLM) | 10 | 70.0 [40.0, 90.0] | 100.0 [100.0, 100.0] | 50.0 [20.0, 80.0] | 0 | 0 | 100.0 | 0.0 |
| (a) single model, first lines (truncation) | 2 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 3,796 | 3,773 | 100.0 | 6.0 |
| (b) single model, strong selection | 2 | 50.0 [0.0, 100.0] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 4,104 | 4,081 | 100.0 | 2.3 |
| (f) single model, long context, same selection | 2 | 50.0 [0.0, 100.0] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 8,446 | 8,424 | 100.0 | 8.2 |
| (d) team, typed JSON messages | 2 | 0.0 [0.0, 0.0] | 50.0 [0.0, 100.0] | 0.0 [0.0, 0.0] | 14,746 | 1,227 | 100.0 | 39.5 |
| (e) team, free-text messages | 2 | 0.0 [0.0, 0.0] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 13,164 | 1,172 | 100.0 | 29.3 |

'Valid' is the share of decisions the model answered in a usable form; for LLM-specialist teams it is the share of specialist messages that passed the schema (typed) or were non-empty (free text).

### Paired comparisons: ollama:qwen2.5-coder:14b

Top-1 difference a - b in points on the incidents both have, paired bootstrap 95% CI, Wilcoxon signed-rank p, Holm-adjusted over this table.

| a | b | n | Difference | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|---|
| team_typed | single_head | 2 | +0.0 | [+0.0, +0.0] | 1 | 1 |
| team_typed | single_selected | 2 | -50.0 | [-100.0, +0.0] | 1 | 1 |
| team_typed | oracle_selected | 2 | -50.0 | [-100.0, +0.0] | 1 | 1 |
| team_typed | heur_baro_metrics | 2 | -100.0 | [-100.0, -100.0] | 0.5 | 1 |
| team_free | single_selected | 2 | -50.0 | [-100.0, +0.0] | 1 | 1 |
| team_free | team_typed | 2 | +0.0 | [+0.0, +0.0] | 1 | 1 |
| single_selected | heur_baro_metrics | 2 | -50.0 | [-100.0, +0.0] | 1 | 1 |

### G4 verdict: ollama:qwen2.5-coder:14b

**KILL.** single_selected is within the CI of the team (difference -50.0 points, CI [-100.0, 0.0] contains 0). Team (d) top-1 0.0, best single model `single_selected` 50.0, difference -50.0 points (95% CI [-100.0, +0.0], 2 incidents). Single-model top-1 on these incidents: single_head 0.0, single_selected 50.0.

### Message ablations: ollama:qwen2.5-coder:14b

One specialist's message is dropped (replaced by an empty report) or replaced by the same service's message from another incident (shuffle). 'root' is the true root-cause service's specialist, 'other' a random other one. A centre that uses the messages loses accuracy when the root's message is ablated and little when another's is.

| Team | Ablation | n | Intact top-1 | Ablated top-1 | Change | 95% CI |
|---|---|---|---|---|---|---|
| team_typed | drop_root | 2 | 0.0 | 0.0 | +0.0 | [+0.0, +0.0] |
| team_typed | shuffle_root | 2 | 0.0 | 0.0 | +0.0 | [+0.0, +0.0] |
| team_typed | drop_other | 2 | 0.0 | 0.0 | +0.0 | [+0.0, +0.0] |
| team_typed | shuffle_other | 2 | 0.0 | 0.0 | +0.0 | [+0.0, +0.0] |
| team_free | drop_root | 2 | 0.0 | 0.0 | +0.0 | [+0.0, +0.0] |
| team_free | shuffle_root | 2 | 0.0 | 0.0 | +0.0 | [+0.0, +0.0] |
| team_free | drop_other | 2 | 0.0 | 0.0 | +0.0 | [+0.0, +0.0] |
| team_free | shuffle_other | 2 | 0.0 | 0.0 | +0.0 | [+0.0, +0.0] |

### Top-1 by system and fault type: ollama:qwen2.5-coder:14b

| Group | n | single_selected | team_typed | team_free | oracle_selected |
|---|---|---|---|---|---|
| fault cpu | 5 | 0.0 | 0.0 | 0.0 | 0.0 |
| fault loss | 5 | 100.0 | 0.0 | 0.0 | 100.0 |
| system SS | 10 | 50.0 | 0.0 | 0.0 | 50.0 |

## Results: qwen3-1.7b

Per incident, percent with 95% bootstrap CI over incidents. Chance for top-1 is 6.7 (1 over the number of services), for fault type 16.7. Tokens are the model tokens read and written per incident across all calls; 'centre prompt' is the largest single prompt.

| Condition | n | Top-1 | Top-3 | Fault type | Tokens / incident | Centre prompt | Valid | Latency (s) |
|---|---|---|---|---|---|---|---|---|
| (c) BARO on log counts (non-LLM) | 10 | 10.0 [0.0, 30.0] | 50.0 [20.0, 80.0] | 10.0 [0.0, 30.0] | 0 | 0 | 100.0 | 0.0 |
| (c) template-surprise ranking (non-LLM) | 10 | 0.0 [0.0, 0.0] | 60.0 [30.0, 90.0] | 0.0 [0.0, 0.0] | 0 | 0 | 100.0 | 0.0 |
| (c) BARO on service metrics (non-LLM) | 10 | 70.0 [40.0, 90.0] | 100.0 [100.0, 100.0] | 50.0 [20.0, 80.0] | 0 | 0 | 100.0 | 0.0 |
| (a) single model, first lines (truncation) | 10 | 30.0 [0.0, 60.0] | 50.0 [20.0, 80.0] | 0.0 [0.0, 0.0] | 3,920 | 3,920 | 100.0 | 2.2 |
| (a) single model, last lines (truncation) | 10 | 0.0 [0.0, 0.0] | 40.0 [10.0, 70.0] | 0.0 [0.0, 0.0] | 3,914 | 3,914 | 100.0 | 2.2 |
| (b) single model, strong selection | 10 | 40.0 [10.0, 70.0] | 60.0 [30.0, 90.0] | 20.0 [0.0, 50.0] | 3,992 | 3,992 | 100.0 | 2.3 |
| (f) single model, long context, same selection | 10 | 30.0 [0.0, 60.0] | 40.0 [10.0, 70.0] | 30.0 [0.0, 60.0] | 32,617 | 32,617 | 100.0 | 12.8 |
| (d) team, typed JSON messages | 10 | 10.0 [0.0, 30.0] | 30.0 [0.0, 60.0] | 30.0 [0.0, 60.0] | 16,054 | 1,481 | 78.0 | 24.8 |
| (e) team, free-text messages | 10 | 20.0 [0.0, 50.0] | 20.0 [0.0, 50.0] | 20.0 [0.0, 50.0] | 15,044 | 1,866 | 100.0 | 22.1 |
| (d') team, typed messages computed by code | 10 | 20.0 [0.0, 50.0] | 50.0 [20.0, 80.0] | 40.0 [10.0, 70.0] | 1,040 | 1,040 | 100.0 | 0.3 |

'Valid' is the share of decisions the model answered in a usable form; for LLM-specialist teams it is the share of specialist messages that passed the schema (typed) or were non-empty (free text).

### Paired comparisons: qwen3-1.7b

Top-1 difference a - b in points on the incidents both have, paired bootstrap 95% CI, Wilcoxon signed-rank p, Holm-adjusted over this table.

| a | b | n | Difference | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|---|
| team_typed | single_head | 10 | -20.0 | [-50.0, +0.0] | 0.5 | 1 |
| team_typed | single_tail | 10 | +10.0 | [+0.0, +30.0] | 1 | 1 |
| team_typed | single_selected | 10 | -30.0 | [-60.0, +0.0] | 0.25 | 1 |
| team_typed | oracle_selected | 10 | -20.0 | [-50.0, +0.0] | 0.5 | 1 |
| team_typed | heur_baro_metrics | 10 | -60.0 | [-90.0, -30.0] | 0.0312 | 0.312 |
| team_free | single_selected | 10 | -20.0 | [-50.0, +0.0] | 0.5 | 1 |
| team_free | team_typed | 10 | +10.0 | [+0.0, +30.0] | 1 | 1 |
| team_typed_heur | team_typed | 10 | +10.0 | [+0.0, +30.0] | 1 | 1 |
| team_typed_heur | single_selected | 10 | -20.0 | [-50.0, +0.0] | 0.5 | 1 |
| single_selected | heur_baro_metrics | 10 | -30.0 | [-60.0, +0.0] | 0.25 | 1 |

### G4 verdict: qwen3-1.7b

**KILL.** single_selected is within the CI of the team (difference -30.0 points, CI [-60.0, 0.0] contains 0). Team (d) top-1 10.0, best single model `single_selected` 40.0, difference -30.0 points (95% CI [-60.0, +0.0], 10 incidents). Single-model top-1 on these incidents: single_head 30.0, single_tail 0.0, single_selected 40.0.

### Message ablations: qwen3-1.7b

One specialist's message is dropped (replaced by an empty report) or replaced by the same service's message from another incident (shuffle). 'root' is the true root-cause service's specialist, 'other' a random other one. A centre that uses the messages loses accuracy when the root's message is ablated and little when another's is.

| Team | Ablation | n | Intact top-1 | Ablated top-1 | Change | 95% CI |
|---|---|---|---|---|---|---|
| team_typed | drop_root | 10 | 10.0 | 10.0 | +0.0 | [+0.0, +0.0] |
| team_typed | shuffle_root | 10 | 10.0 | 20.0 | +10.0 | [+0.0, +30.0] |
| team_typed | drop_other | 10 | 10.0 | 10.0 | +0.0 | [+0.0, +0.0] |
| team_typed | shuffle_other | 10 | 10.0 | 10.0 | +0.0 | [+0.0, +0.0] |
| team_free | drop_root | 10 | 20.0 | 0.0 | -20.0 | [-50.0, +0.0] |
| team_free | shuffle_root | 10 | 20.0 | 20.0 | +0.0 | [+0.0, +0.0] |
| team_free | drop_other | 10 | 20.0 | 20.0 | +0.0 | [+0.0, +0.0] |
| team_free | shuffle_other | 10 | 20.0 | 20.0 | +0.0 | [+0.0, +0.0] |

### Top-1 by system and fault type: qwen3-1.7b

| Group | n | single_selected | team_typed | team_free | oracle_selected |
|---|---|---|---|---|---|
| fault cpu | 5 | 40.0 | 0.0 | 20.0 | 40.0 |
| fault loss | 5 | 40.0 | 20.0 | 20.0 | 20.0 |
| system SS | 10 | 40.0 | 10.0 | 20.0 | 30.0 |

## Setup

- **Data:** RCAEval RE2 (Pham et al., WWW 2025 Companion, arXiv:2412.17015; Zenodo 10.5281/zenodo.14590730, CC-BY-4.0), Sock Shop (SS) and Online Boutique (OB). Each incident is one fault injection (a root-cause service and a fault type among cpu, mem, disk, delay, loss, socket) with per-service logs and service metrics; the alert is the injection time. Repetition 1 of every case is the dev split (prompt design, smoke runs); the reported numbers use the test split (repetitions 2 and 3). Incidents whose root service appears in neither logs nor metrics are dropped.
- **Design change forced by the data:** the task was meant to be log triage, but the non-LLM log baselines (BARO on log counts, template surprise) are near chance on the dev split, so logs alone cannot identify the root cause in RE2. Each service is therefore a source of its logs **and its metrics** (`simple_metrics.csv`: cpu, mem, socket, diskio, workload, latency, error, summarised as robust deviations from the pre-alert range). Candidates are all services that appear in either, so infrastructure services (databases, queue) are candidates too. The metric summary is available to every condition, so the strong single-model selection (b) sees it as well. Logs-only results for the non-LLM baselines are in the table (rows marked logs).
- **Conditions:** see `experiments/log_triage.py`. All centre prompts are at most 4,096 tokens, instructions and options included; specialists read at most 3,072 tokens of their own stream and send at most 128 tokens; the long-context oracle (f) reads 32,768 tokens (8,192 for the Ollama model).
- **Scoring:** the root-cause service is ranked by next-token log-probabilities over option letters (Qwen3), or by a structured JSON reply (Ollama); fault type is the best letter given the chosen service.
- **Not done in step 1:** no LoRA training of specialists or centre (step 2), no learned compression of messages, no KV handoff (G5).
