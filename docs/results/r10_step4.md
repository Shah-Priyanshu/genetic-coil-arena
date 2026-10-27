# R10 step 4: LLM log specialists on RCAEval RE3 (goal G4d)

Built on the PC from the shard results of jobs `r10s4-gate`, `r10s4-o14-*` and `r10s4-q3-*` (pinned commit 34a24be2; the report code is unchanged since), while the runner's `r10s4-zz-aggregate` job was still waiting for a CPU slot.

**G4d verdict: KILL**

Best non-LLM system on dev: nonllm_fusion (test top-1 80.6); best on the test split itself: nonllm_fusion (80.6).

## Data (RCAEval RE3)

| system | incidents | cases | root services | candidate services | faults | median log lines |
|---|---|---|---|---|---|---|
| RE3-OB | 30 | 10 | adservice, cartservice, currencyservice, emailservice | 12 | f1:9, f2:3, f3:6, f4:6, f5:6 | 70,193 |
| RE3-SS | 30 | 9 | carts, front-end, orders | 15 | f1:10, f2:3, f3:10, f4:7 | 84,302 |
| RE3-TT | 30 | 9 | ts-auth-service, ts-route-service | 68 | f1:7, f2:7, f3:7, f3_1:3, f4:6 | 56,728 |

## Gate: do the logs carry root-cause signal?

| system | split | root has a new warning/error template | mean such templates, root | mean, other services | root strictly first by that count |
|---|---|---|---|---|---|
| RE3-OB | all | 60% | 22.2 | 6.27 | 43% |
| RE3-SS | all | 100% | 5.7 | 0.12 | 100% |
| RE3-TT | all | 100% | 3.7 | 0.13 | 77% |
| RE3-OB | test | 60% | 21.4 | 6.18 | 40% |
| RE3-SS | test | 100% | 5.7 | 0.14 | 100% |
| RE3-TT | test | 100% | 3.7 | 0.14 | 76% |

## Non-LLM systems (root-cause top-1, %)

| system | dev/SS | dev/OB | dev/TT | dev/all | test/SS | test/OB | test/TT | test/all |
|---|---|---|---|---|---|---|---|---|
| BARO on service metrics | 78 | 70 | 44 | 64 | 71 | 60 | 38 | 56 |
| BARO on log counts | 33 | 10 | 89 | 43 | 29 | 10 | 90 | 44 |
| template-surprise ranking | 33 | 0 | 89 | 39 | 29 | 0 | 86 | 39 |
| new warning/error templates (count) | 100 | 50 | 78 | 75 | 100 | 45 | 76 | 74 |
| new warning/error templates (severity x log count) | 100 | 40 | 78 | 71 | 86 | 30 | 52 | 56 |
| new warning/error templates (discounted by spread) | 100 | 40 | 78 | 71 | 86 | 30 | 62 | 60 |
| all new templates (severity x log count) | 33 | 0 | 100 | 43 | 29 | 0 | 100 | 44 |
| new warning/error/lifecycle templates (count) | 100 | 50 | 89 | 79 | 100 | 50 | 81 | 77 |
| reciprocal-rank fusion of metric and log scores (dev-fitted) | 100 | 70 | 89 | 86 | 95 | 70 | 76 | 81 |

## Main results (test split, root-cause top-1 and top-3, %)

| system | n | top-1 [95% CI] | top-3 | tokens | bytes |
|---|---|---|---|---|---|
| BARO on service metrics (nonllm_baro_metrics) | 62 | 56.5 [43.5, 69.4] | 90.3 | 0 | 0 |
| BARO on log counts (nonllm_baro_logs) | 62 | 43.5 [32.3, 54.8] | 82.3 | 0 | 0 |
| template-surprise ranking (nonllm_surprise_logs) | 62 | 38.7 [27.4, 51.6] | 64.5 | 0 | 0 |
| new warning/error templates (count) (nonllm_new_err) | 62 | 74.2 [62.9, 83.9] | 79.0 | 0 | 0 |
| new warning/error templates (severity x log count) (nonllm_new_err_mass) | 62 | 56.5 [43.5, 67.7] | 82.3 | 0 | 0 |
| new warning/error templates (discounted by spread) (nonllm_new_err_idf) | 62 | 59.7 [48.4, 71.0] | 83.9 | 0 | 0 |
| all new templates (severity x log count) (nonllm_new_any_mass) | 62 | 43.5 [30.6, 56.5] | 77.4 | 0 | 0 |
| new warning/error/lifecycle templates (count) (nonllm_new_ext) | 62 | 77.4 [66.1, 87.1] | 91.9 | 0 | 0 |
| reciprocal-rank fusion of metric and log scores (dev-fitted) (nonllm_fusion) | 62 | 80.6 [71.0, 90.3] | 93.5 | 0 | 0 |
| team: metric + LLM log specialists, rule centre (team4_rule[qwen2.5-coder:14b]) | 62 | 62.9 [50.0, 74.2] | 88.7 | 23,592 | 6,523 |
| team: metric + LLM log specialists, rule centre (team4_rule[qwen3-1.7b]) | 62 | 50.0 [37.1, 62.9] | 61.3 | 25,633 | 6,013 |
| team: metric + LLM log specialists, LLM centre (team4_llm[qwen2.5-coder:14b]) | 62 | 64.5 [51.6, 75.8] | 87.1 | 24,949 | 6,523 |
| single LLM, selected view of logs + metrics (single_all[qwen2.5-coder:14b]) | 62 | 62.9 [51.6, 74.2] | 85.5 | 6,045 | 0 |
| single LLM, selected view of logs only (single_logs[qwen2.5-coder:14b]) | 62 | 50.0 [37.1, 62.9] | 80.6 | 6,042 | 0 |

## G4d comparisons (paired, test split)

| LLM team | rule | top-1 | vs dev-best non-LLM [95% CI] | p (Holm) | vs test-best non-LLM [95% CI] | verdict |
|---|---|---|---|---|---|---|
| team4_rule[qwen2.5-coder:14b] | w2 | 62.9 | -17.7 [-30.6, -4.8] | 0.015 | -17.7 [-30.6, -6.5] | KILL |
| team4_llm[qwen2.5-coder:14b] | - | 64.5 | -16.1 [-29.0, -4.8] | 0.015 | -16.1 [-27.4, -4.8] | KILL |
| team4_rule[qwen3-1.7b] | w0.5 | 50.0 | -30.6 [-43.5, -17.7] | <0.001 | -30.6 [-45.2, -16.1] | KILL |

## Team against the best single LLM

| team | best single | single top-1 | team - single [95% CI] |
|---|---|---|---|
| team4_rule[qwen2.5-coder:14b] | single_all[qwen2.5-coder:14b] | 62.9 | +0.0 [-11.3, +11.3] |
| team4_llm[qwen2.5-coder:14b] | single_all[qwen2.5-coder:14b] | 62.9 | +1.6 [-8.1, +12.9] |
| team4_rule[qwen3-1.7b] | single_all[qwen2.5-coder:14b] | 62.9 | -12.9 [-25.8, +0.0] |

## Ablations (test split)

| system | ablation | intact top-1 | ablated top-1 | change [95% CI] |
|---|---|---|---|---|
| team4_llm[qwen2.5-coder:14b] | drop_root | 64.5 | 16.1 | -48.4 [-61.3, -35.5] |
| team4_llm[qwen2.5-coder:14b] | shuffle_root | 66.7 | 35.0 | -31.7 [-45.0, -18.3] |
| team4_rule[qwen2.5-coder:14b] | drop_other | 62.9 | 62.9 | +0.0 [+0.0, +0.0] |
| team4_rule[qwen2.5-coder:14b] | drop_root | 62.9 | 0.0 | -62.9 [-74.2, -50.0] |
| team4_rule[qwen2.5-coder:14b] | shuffle_other | 63.9 | 63.9 | +0.0 [+0.0, +0.0] |
| team4_rule[qwen2.5-coder:14b] | shuffle_root | 65.0 | 23.3 | -41.7 [-55.0, -28.3] |
| team4_rule[qwen3-1.7b] | drop_other | 50.0 | 50.0 | +0.0 [+0.0, +0.0] |
| team4_rule[qwen3-1.7b] | drop_root | 50.0 | 50.0 | +0.0 [+0.0, +0.0] |
| team4_rule[qwen3-1.7b] | shuffle_other | 52.9 | 50.0 | -2.9 [-8.8, +0.0] |
| team4_rule[qwen3-1.7b] | shuffle_root | 50.0 | 61.1 | +11.1 [+2.8, +22.2] |

## Top-1 per RE3 system (test split, %)

| system | RE3-OB | RE3-SS | RE3-TT |
|---|---|---|---|
| nonllm_baro_metrics | 60 | 71 | 38 |
| nonllm_baro_logs | 10 | 29 | 90 |
| nonllm_surprise_logs | 0 | 29 | 86 |
| nonllm_new_err | 45 | 100 | 76 |
| nonllm_new_err_mass | 30 | 86 | 52 |
| nonllm_new_err_idf | 30 | 86 | 62 |
| nonllm_new_any_mass | 0 | 29 | 100 |
| nonllm_new_ext | 50 | 100 | 81 |
| nonllm_fusion | 70 | 95 | 76 |
| team4_rule[qwen2.5-coder:14b] | 70 | 76 | 43 |
| team4_rule[qwen3-1.7b] | 55 | 67 | 29 |
| team4_llm[qwen2.5-coder:14b] | 60 | 71 | 62 |
| single_all[qwen2.5-coder:14b] | 70 | 76 | 43 |
| single_logs[qwen2.5-coder:14b] | 40 | 76 | 33 |

## Rule-centre weighting chosen on dev

| log specialist | n dev | dev top-1 by weighting | chosen |
|---|---|---|---|
| qwen2.5-coder:14b | 28 | log_only 54, w0.5 71, w1 75, w2 79, lex 79 | w2 |
| qwen3-1.7b | 28 | log_only 0, w0.5 64, w1 36, w2 29, lex 29 | w0.5 |

## Analysis

<!-- analysis:start -->
- **G4d is a kill, and not a close one.** Every LLM team is significantly *below* the dev-chosen non-LLM fusion (80.6 top-1), the opposite of the +5 the goal asked for.
  - 14B, LLM centre: 64.5, −16.1 [−29.0, −4.8].
  - 14B, rule centre: 62.9, −17.7 [−30.6, −4.8].
  - Qwen3-1.7B, rule centre: 50.0, −30.6 [−43.5, −17.7].
  - The exploratory gemma4:e4b run agrees: 59.7 with a rule centre and 54.8 with an LLM centre (`r10_step4_explore.md`).
- **The logs do carry the signal; the LLMs read it worse than a count does.** A plain count of new warning/error/lifecycle templates after onset reaches 77.4 top-1 with no model. Every LLM condition is below it. On dev, the 14B's log reports alone rank the root first in 54% of incidents, against 79% for that count.
- **The 14B finds the root but cannot rank it.** Measured from its test-split reports (`logmsgs_*.jsonl`):
  - It flags the root service (suspicion > 0) in 100% of Sock Shop and Train Ticket incidents and 80% of Online Boutique ones.
  - It also flags 41.5% of the other services on Sock Shop, 32.7% on Online Boutique and 20.3% on Train Ticket.
  - So the root's suspicion is strictly highest in only 43% (SS), 60% (OB) and 14% (TT) of incidents. On Train Ticket's 68 services, an average of 11.5 other services score at or above the root.
  - Suspicion is a saturated number: 88% of the 14B's non-zero suspicions (over all 90 incidents) are exactly 0.8, and 10% are 0.9. It does not separate a new error in the faulty service from routine errors elsewhere. The template count does, because it compares against the pre-onset baseline.
- **Splitting the logs across specialists did not help either.** The 14B team equals one 14B reading a selected view of all services and metrics at the same budget (+0.0 [−11.3, +11.3] with the rule centre, +1.6 [−8.1, +12.9] with the LLM centre).
- **The reports drive the team's decision.**
  - 14B rule team (weights chosen on dev: logs ×2, metrics ×1): dropping the root's log report gives 0.0. With that weighting, any other service the specialist flags outranks a root that has only its metric score. Shuffling the root's report in from an incident where it was healthy gives 23.3.
  - 14B LLM centre: dropping the root's report gives 16.1, shuffling 35.0.
  - Qwen3-1.7B: its reports are anti-informative. Logs alone rank the root first in 0% of dev incidents, so dev chose the lowest log weight. Dropping the root's report changes nothing, and replacing it with a healthy-incident report *raises* top-1 by +11.1 [+2.8, +22.2].
- **Where the LLM team is competitive.** On Online Boutique, where the gate found the logs weakest (the root has a new warning/error template in only 60% of incidents), the 14B rule team ties the fusion at 70. There the metrics carry the decision.
- **What this means for the roadmap.** This is the third R10 test of LLM log specialists and the third kill (G4b on RE2, the G4c LoRA on RE2, now G4d on RE3), this time on data where the logs carry the signal. The specialists that have earned a place are code and small trained models (BARO-style metric scores, the G4c fault classifier, the template count). The LLMs that have earned a place are at the language edges: the translator (G3) and the summariser (G6). The next R10 work should use the trained team as the core and test it end to end behind the translator, not try more LLM specialists.
- **Caveats.**
  - 62 test incidents over 3 systems, and 9–10 cases per system repeated across repetitions, so the CIs are wide (about ±12 points).
  - Zero-shot prompts only. The prompt was fixed after a 3-incident smoke on OB and not tuned on dev. A prompt that hands the model the pre-onset templates, or asks for a count, could close part of the gap, but it would then be re-implementing the count.
  - The 14B and 1.7B were tested at one context budget (1,500-token log view per service).
<!-- analysis:end -->
