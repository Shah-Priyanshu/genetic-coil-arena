# R10 step 2: a heterogeneous team with typed messages (goal G4b)

Step 2 of R10 on RCAEval RE2 (Sock Shop + Online Boutique). Metric specialists (code, one per service, own metrics only) send typed anomaly reports; log specialists (a model, one per service, own logs around the reported onset) send typed fault hypotheses; a rule or LLM centre decides. Numbers are on the test split (120 incidents); the dev split only chose the rule centre's fault policy.

## Verdict

**G4b: KILL.** Pass: a hybrid team beats the best single system on joint accuracy by at least 5 points (95% CI excludes 0, Holm-adjusted p < 0.05) with root-cause top-1 non-inferior (CI lower bound above -3 points). Kill: the best single system is within the CI on joint accuracy.

| Hybrid team | Best single (joint) | Team joint | Single joint | Difference [95% CI] | p (Holm) | Top-1 vs it | Top-1 vs best-top-1 single | Verdict |
|---|---|---|---|---|---|---|---|---|
| team_hybrid_rule[qwen2.5-coder:14b] | baro_metrics | 45.0 | 45.0 | +0.0 [+0.0, +0.0] | 1 | +0.0 [+0.0, +0.0] | baro_metrics: +0.0 [+0.0, +0.0] | **KILL** |
| team_hybrid_rule[qwen3-1.7b] | baro_metrics | 43.3 | 45.0 | -1.7 [-5.0, +1.7] | 0.635 | +0.0 [+0.0, +0.0] | baro_metrics: +0.0 [+0.0, +0.0] | **KILL** |
| team_hybrid_llm[qwen2.5-coder:14b] | baro_metrics | 33.3 | 45.0 | -11.7 [-20.8, -2.5] | 0.0589 | +0.0 [-5.8, +6.7] | baro_metrics: +0.0 [-5.8, +5.8] | **KILL** |
| team_hybrid_llm[qwen3-1.7b] | baro_metrics | 33.3 | 45.0 | -11.7 [-20.0, -3.3] | 0.0423 | +2.5 [-2.5, +8.3] | baro_metrics: +2.5 [-2.5, +8.3] | **KILL** |

Honest reading: BARO is decomposable per service, so a metric-only team with a rule centre is distributed BARO and cannot beat BARO (check below); the real test is the hybrid's fault-type gain from the log specialists, at unchanged root-cause accuracy. Check: the metric-only rule team names the same top service as BARO on 100.0% and the same fault on 100.0% of 120 incidents.

## Analysis

<!-- analysis:start -->
- **The logs added nothing.** No hybrid team beats BARO on joint accuracy. With the 14B log specialists, the rule centre's dev-chosen policy ignores the log hypotheses (`metric`), so the team is BARO exactly. With Qwen3-1.7B the policy follows the logs and loses 1.7 points [-5.0, +1.7].
- **Log reports hurt an LLM centre.** Adding them to the 14B centre cuts joint accuracy by 13.3 points [-22.5, -4.2] (fault type 50.0 to 36.7) at unchanged top-1. The zero-shot hypotheses are noisy and the centre trusts them.
- **The typed metric messages do carry the signal.**
  - The 14B centre reading about 800 tokens of typed metric reports reaches 90.0 top-1 and 46.7 joint [38.3, 55.8]. The same model reading 4,000 tokens of selected raw data reaches 70.0 and 29.2 [21.7, 37.5]; this pair is not in the paired table.
  - In step 1, the typed team without metric scores got 4.2% top-1.
  - The ablations show that the centres rely on the messages: dropping the root service's messages takes top-1 to 0, and shuffling them takes it to about 33.
- **Fault type is the bottleneck.** Every system's joint accuracy is near 0 to 20% on disk, loss and socket faults, and 75 to 85% on cpu, delay and mem. Either the per-service metric maxima do not separate those faults, or zero-shot log readers cannot find the evidence. A trained fault-type specialist is the obvious next test.
- **Cost:** the 14B hybrid spends 10.6k tokens and 17.7 s per incident to match a rule that costs nothing.
<!-- analysis:end -->

## Results (test split)

Percent per incident with 95% bootstrap CI. Joint = root-cause service and fault type both right. Chance for top-1 is about 7.5, for fault type 16.7, for joint about 1.3. Tokens are model tokens read and written per incident over all calls (specialists and centre); bytes are the typed messages' compact JSON; the rule centre's fault policy per model was chosen on dev.

| System | n | Top-1 | Top-3 | Fault type | Joint | Tokens / incident | Centre prompt | Bytes / incident | Latency (s) |
|---|---|---|---|---|---|---|---|---|---|
| baro_metrics | 120 | 88.3 [82.5, 93.3] | 99.2 [97.5, 100.0] | 48.3 [39.2, 57.5] | 45.0 [35.8, 53.4] | 0 | 0 | 0 | 0.0 |
| single_selected[qwen2.5-coder:14b] | 120 | 70.0 [61.7, 78.3] | 90.0 [84.2, 95.0] | 44.2 [35.8, 53.3] | 29.2 [21.7, 37.5] | 4,010 | 3,987 | 0 | 15.6 |
| single_selected[qwen3-1.7b] | 120 | 64.2 [55.8, 72.5] | 79.2 [71.7, 86.7] | 31.7 [23.3, 40.0] | 16.7 [10.0, 23.3] | 3,991 | 3,991 | 0 | 1.1 |
| oracle_selected[qwen2.5-coder:14b] | 120 | 74.2 [65.8, 81.7] | 91.7 [86.7, 95.8] | 39.2 [30.8, 48.3] | 29.2 [21.7, 37.5] | 8,073 | 8,050 | 0 | 17.5 |
| oracle_selected[qwen3-1.7b] | 120 | 56.7 [47.5, 65.0] | 74.2 [65.8, 81.7] | 29.2 [21.7, 37.5] | 15.8 [10.0, 22.5] | 32,480 | 32,480 | 0 | 12.9 |
| team_metric_rule | 120 | 88.3 [82.5, 93.3] | 99.2 [97.5, 100.0] | 48.3 [39.2, 57.5] | 45.0 [35.8, 53.4] | 0 | 0 | 1,122 | 0.0 |
| team_metric_llm[qwen2.5-coder:14b] | 120 | 90.0 [84.2, 95.0] | 99.2 [97.5, 100.0] | 50.0 [41.7, 59.2] | 46.7 [38.3, 55.8] | 836 | 813 | 1,122 | 1.1 |
| team_hybrid_rule[qwen2.5-coder:14b] | 120 | 88.3 [82.5, 93.3] | 99.2 [97.5, 100.0] | 48.3 [39.2, 57.5] | 45.0 [35.8, 53.4] | 10,626 | 0 | 2,432 | 17.7 |
| team_hybrid_rule[qwen3-1.7b] | 120 | 88.3 [82.5, 93.3] | 99.2 [97.5, 100.0] | 46.7 [37.5, 55.8] | 43.3 [35.0, 52.5] | 11,285 | 0 | 3,229 | 34.1 |
| team_hybrid_llm[qwen2.5-coder:14b] | 120 | 88.3 [82.5, 93.3] | 97.5 [94.2, 100.0] | 36.7 [28.3, 45.0] | 33.3 [25.0, 41.7] | 11,819 | 1,170 | 2,432 | 18.9 |
| team_hybrid_llm[qwen3-1.7b] | 120 | 90.8 [85.8, 95.8] | 97.5 [94.2, 100.0] | 35.8 [27.5, 44.2] | 33.3 [25.8, 41.7] | 12,728 | 1,420 | 3,229 | 35.4 |

### Fault policy of the rule centre (chosen on dev)

| Log-specialist model | Dev incidents | metric | log | log_c50 | log_c80 | Chosen |
|---|---|---|---|---|---|---|
| qwen3-1.7b | 60 | 38.3 | 41.7 | 41.7 | 41.7 | log |
| qwen2.5-coder:14b | 60 | 38.3 | 36.7 | 36.7 | 36.7 | metric |

### Paired comparisons of joint accuracy

Difference a - b in points on shared test incidents, paired bootstrap 95% CI, Wilcoxon p, Holm over this table.

| a | b | n | Difference | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|---|
| team_hybrid_rule[qwen2.5-coder:14b] | baro_metrics | 120 | +0.0 | [+0.0, +0.0] | 1 | 1 |
| team_hybrid_rule[qwen2.5-coder:14b] | single_selected[qwen2.5-coder:14b] | 120 | +15.8 | [+8.3, +24.2] | 0.000256 | 0.00537 |
| team_hybrid_rule[qwen2.5-coder:14b] | single_selected[qwen3-1.7b] | 120 | +28.3 | [+18.3, +38.3] | 9.23e-07 | 2.21e-05 |
| team_hybrid_rule[qwen2.5-coder:14b] | oracle_selected[qwen2.5-coder:14b] | 120 | +15.8 | [+8.3, +24.2] | 0.000256 | 0.00537 |
| team_hybrid_rule[qwen2.5-coder:14b] | oracle_selected[qwen3-1.7b] | 120 | +29.2 | [+19.2, +39.2] | 5.73e-07 | 1.43e-05 |
| team_hybrid_rule[qwen2.5-coder:14b] | team_metric_rule | 120 | +0.0 | [+0.0, +0.0] | 1 | 1 |
| team_hybrid_rule[qwen3-1.7b] | baro_metrics | 120 | -1.7 | [-5.0, +1.7] | 0.317 | 1 |
| team_hybrid_rule[qwen3-1.7b] | single_selected[qwen2.5-coder:14b] | 120 | +14.2 | [+6.7, +22.5] | 0.000674 | 0.0128 |
| team_hybrid_rule[qwen3-1.7b] | single_selected[qwen3-1.7b] | 120 | +26.7 | [+16.7, +36.7] | 2.38e-06 | 5.47e-05 |
| team_hybrid_rule[qwen3-1.7b] | oracle_selected[qwen2.5-coder:14b] | 120 | +14.2 | [+6.7, +21.7] | 0.000674 | 0.0128 |
| team_hybrid_rule[qwen3-1.7b] | oracle_selected[qwen3-1.7b] | 120 | +27.5 | [+16.7, +37.5] | 2.43e-06 | 5.47e-05 |
| team_hybrid_rule[qwen3-1.7b] | team_metric_rule | 120 | -1.7 | [-5.0, +1.7] | 0.317 | 1 |
| team_hybrid_llm[qwen2.5-coder:14b] | baro_metrics | 120 | -11.7 | [-21.7, -2.5] | 0.0196 | 0.216 |
| team_hybrid_llm[qwen2.5-coder:14b] | single_selected[qwen2.5-coder:14b] | 120 | +4.2 | [-4.2, +12.5] | 0.317 | 1 |
| team_hybrid_llm[qwen2.5-coder:14b] | single_selected[qwen3-1.7b] | 120 | +16.7 | [+6.7, +26.7] | 0.00157 | 0.0219 |
| team_hybrid_llm[qwen2.5-coder:14b] | oracle_selected[qwen2.5-coder:14b] | 120 | +4.2 | [-4.2, +12.5] | 0.336 | 1 |
| team_hybrid_llm[qwen2.5-coder:14b] | oracle_selected[qwen3-1.7b] | 120 | +17.5 | [+7.5, +27.5] | 0.00104 | 0.0156 |
| team_hybrid_llm[qwen2.5-coder:14b] | team_metric_rule | 120 | -11.7 | [-21.7, -2.5] | 0.0196 | 0.216 |
| team_hybrid_llm[qwen3-1.7b] | baro_metrics | 120 | -11.7 | [-20.0, -3.3] | 0.0106 | 0.138 |
| team_hybrid_llm[qwen3-1.7b] | single_selected[qwen2.5-coder:14b] | 120 | +4.2 | [-3.3, +11.7] | 0.297 | 1 |
| team_hybrid_llm[qwen3-1.7b] | single_selected[qwen3-1.7b] | 120 | +16.7 | [+7.5, +25.8] | 0.000858 | 0.0137 |
| team_hybrid_llm[qwen3-1.7b] | oracle_selected[qwen2.5-coder:14b] | 120 | +4.2 | [-4.2, +12.5] | 0.336 | 1 |
| team_hybrid_llm[qwen3-1.7b] | oracle_selected[qwen3-1.7b] | 120 | +17.5 | [+7.5, +27.5] | 0.000772 | 0.0131 |
| team_hybrid_llm[qwen3-1.7b] | team_metric_rule | 120 | -11.7 | [-20.0, -3.3] | 0.0106 | 0.138 |
| team_metric_rule | baro_metrics | 120 | +0.0 | [+0.0, +0.0] | 1 | 1 |

### Dropping the log specialists

Each hybrid team against the metric-only team with the same centre.

| Hybrid team | Without logs | Metric | n | Difference | 95% CI |
|---|---|---|---|---|---|
| team_hybrid_rule[qwen2.5-coder:14b] | team_metric_rule | joint | 120 | +0.0 | [+0.0, +0.0] |
| team_hybrid_rule[qwen2.5-coder:14b] | team_metric_rule | fault | 120 | +0.0 | [+0.0, +0.0] |
| team_hybrid_rule[qwen2.5-coder:14b] | team_metric_rule | top1 | 120 | +0.0 | [+0.0, +0.0] |
| team_hybrid_rule[qwen3-1.7b] | team_metric_rule | joint | 120 | -1.7 | [-5.0, +1.7] |
| team_hybrid_rule[qwen3-1.7b] | team_metric_rule | fault | 120 | -1.7 | [-5.0, +1.7] |
| team_hybrid_rule[qwen3-1.7b] | team_metric_rule | top1 | 120 | +0.0 | [+0.0, +0.0] |
| team_hybrid_llm[qwen2.5-coder:14b] | team_metric_llm[qwen2.5-coder:14b] | joint | 120 | -13.3 | [-22.5, -4.2] |
| team_hybrid_llm[qwen2.5-coder:14b] | team_metric_llm[qwen2.5-coder:14b] | fault | 120 | -13.3 | [-22.5, -4.2] |
| team_hybrid_llm[qwen2.5-coder:14b] | team_metric_llm[qwen2.5-coder:14b] | top1 | 120 | -1.7 | [-6.7, +3.3] |
| team_hybrid_llm[qwen3-1.7b] | team_metric_llm[qwen2.5-coder:14b] | joint | 120 | -13.3 | [-22.5, -4.2] |
| team_hybrid_llm[qwen3-1.7b] | team_metric_llm[qwen2.5-coder:14b] | fault | 120 | -14.2 | [-22.5, -5.8] |
| team_hybrid_llm[qwen3-1.7b] | team_metric_llm[qwen2.5-coder:14b] | top1 | 120 | +0.8 | [-3.3, +5.0] |

### Message ablations

One service's messages (metric and log) are emptied (drop) or replaced by another incident's for the same service (shuffle); 'root' is the true root-cause service, 'other' a random other one. LLM centres are ablated on the root service only.

| Team | Ablation | n | Top-1 intact | Top-1 ablated | Change [95% CI] | Joint intact | Joint ablated | Change [95% CI] |
|---|---|---|---|---|---|---|---|---|
| team_metric_rule | drop_other | 120 | 88.3 | 91.7 | +3.3 [+0.8, +6.7] | 45.0 | 45.0 | +0.0 [+0.0, +0.0] |
| team_metric_rule | drop_root | 120 | 88.3 | 0.0 | -88.3 [-94.2, -82.5] | 45.0 | 0.0 | -45.0 [-53.3, -35.8] |
| team_metric_rule | shuffle_other | 120 | 88.3 | 88.3 | +0.0 [-4.2, +4.2] | 45.0 | 45.0 | +0.0 [+0.0, +0.0] |
| team_metric_rule | shuffle_root | 120 | 88.3 | 32.5 | -55.8 [-65.0, -46.7] | 45.0 | 4.2 | -40.8 [-49.2, -32.5] |
| team_metric_llm[qwen2.5-coder:14b] | drop_root | 120 | 90.0 | 0.8 | -89.2 [-94.2, -83.3] | 46.7 | 0.0 | -46.7 [-55.0, -37.5] |
| team_metric_llm[qwen2.5-coder:14b] | shuffle_root | 120 | 90.0 | 36.7 | -53.3 [-62.5, -43.3] | 46.7 | 5.8 | -40.8 [-49.2, -32.5] |
| team_hybrid_rule[qwen2.5-coder:14b] | drop_other | 120 | 88.3 | 91.7 | +3.3 [+0.8, +6.7] | 45.0 | 45.0 | +0.0 [+0.0, +0.0] |
| team_hybrid_rule[qwen2.5-coder:14b] | drop_root | 120 | 88.3 | 0.0 | -88.3 [-94.2, -82.5] | 45.0 | 0.0 | -45.0 [-53.3, -35.8] |
| team_hybrid_rule[qwen2.5-coder:14b] | shuffle_other | 120 | 88.3 | 88.3 | +0.0 [-4.2, +4.2] | 45.0 | 45.0 | +0.0 [+0.0, +0.0] |
| team_hybrid_rule[qwen2.5-coder:14b] | shuffle_root | 120 | 88.3 | 32.5 | -55.8 [-65.0, -46.7] | 45.0 | 4.2 | -40.8 [-49.2, -32.5] |
| team_hybrid_rule[qwen3-1.7b] | drop_other | 120 | 88.3 | 91.7 | +3.3 [+0.8, +6.7] | 43.3 | 44.2 | +0.8 [+0.0, +2.5] |
| team_hybrid_rule[qwen3-1.7b] | drop_root | 120 | 88.3 | 0.0 | -88.3 [-94.2, -82.5] | 43.3 | 0.0 | -43.3 [-52.5, -35.0] |
| team_hybrid_rule[qwen3-1.7b] | shuffle_other | 120 | 88.3 | 88.3 | +0.0 [-4.2, +4.2] | 43.3 | 44.2 | +0.8 [+0.0, +2.5] |
| team_hybrid_rule[qwen3-1.7b] | shuffle_root | 120 | 88.3 | 32.5 | -55.8 [-65.0, -46.7] | 43.3 | 4.2 | -39.2 [-47.5, -30.8] |
| team_hybrid_llm[qwen2.5-coder:14b] | drop_root | 120 | 88.3 | 0.0 | -88.3 [-93.3, -82.5] | 33.3 | 0.0 | -33.3 [-41.7, -25.0] |
| team_hybrid_llm[qwen2.5-coder:14b] | shuffle_root | 120 | 88.3 | 36.7 | -51.7 [-61.7, -41.7] | 33.3 | 5.0 | -28.3 [-37.5, -19.2] |
| team_hybrid_llm[qwen3-1.7b] | drop_root | 120 | 90.8 | 0.0 | -90.8 [-95.8, -85.0] | 33.3 | 0.0 | -33.3 [-41.7, -25.0] |
| team_hybrid_llm[qwen3-1.7b] | shuffle_root | 120 | 90.8 | 32.5 | -58.3 [-67.5, -48.3] | 33.3 | 5.0 | -28.3 [-36.7, -20.0] |

### Joint accuracy by fault type and system

| Group | n | baro_metrics | single_selected[qwen2.5-coder:14b] | single_selected[qwen3-1.7b] | oracle_selected[qwen2.5-coder:14b] | oracle_selected[qwen3-1.7b] | team_metric_rule | team_metric_llm[qwen2.5-coder:14b] | team_hybrid_rule[qwen2.5-coder:14b] |
|---|---|---|---|---|---|---|---|---|---|
| fault cpu | 20 | 75.0 | 40.0 | 5.0 | 40.0 | 10.0 | 75.0 | 85.0 | 75.0 |
| fault delay | 20 | 85.0 | 55.0 | 40.0 | 45.0 | 35.0 | 85.0 | 75.0 | 85.0 |
| fault disk | 20 | 5.0 | 0.0 | 0.0 | 0.0 | 10.0 | 5.0 | 15.0 | 5.0 |
| fault loss | 20 | 0.0 | 5.0 | 0.0 | 5.0 | 10.0 | 0.0 | 0.0 | 0.0 |
| fault mem | 20 | 85.0 | 55.0 | 15.0 | 65.0 | 10.0 | 85.0 | 85.0 | 85.0 |
| fault socket | 20 | 20.0 | 20.0 | 40.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 |
| system OB | 60 | 45.0 | 43.3 | 23.3 | 41.7 | 25.0 | 45.0 | 51.7 | 45.0 |
| system SS | 60 | 45.0 | 15.0 | 10.0 | 16.7 | 6.7 | 45.0 | 41.7 | 45.0 |

## Setup

- **Data and splits:** as step 1 (`docs/results/r10_log_triage.md`): RCAEval RE2, SS and OB, repetition 1 = dev, repetitions 2 and 3 = test. Singles are re-run here (same code and budgets as step 1: 4,096-token centre prompt, 8,192 / 32,768 oracle context) so every comparison is paired per incident; step 1's tables hold no per-incident records.
- **Metric specialist:** per service, score = largest robust deviation (IQR units) of its metrics after the alert, onset = earliest first-deviation time among its top 3 metrics (z >= 3), top metrics with direction. Typed JSON (`METRIC_SCHEMA`), tens of bytes.
- **Log specialist:** per service, its own templated log lines in a window of -120 s to +300 s around the reported onset, ranked by rate change against the earlier baseline, at most 1,500 tokens; typed JSON (`LOG_SCHEMA`): fault, confidence, up to 3 evidence templates. Zero-shot, no training. A reply that is not valid JSON for the schema becomes an empty report (fault unknown).
- **Centres:** rule = argmax score, fault from the top metric kind (`metric`) or the log hypothesis of that service (`log`, or only when confidence is at least 0.5 / 0.8, else the metric kind); LLM = qwen2.5-coder:14b reading all typed messages (JSON) within the 4,096-token prompt.
- **Object layer:** the hybrid team with the rule centre also ran through `CommandCentre`, envelopes and `TraceLog` on a subset (`docs/results/r10_step2.json`, key `runtime`).
- **Not done:** no training of any specialist or centre, no learned compression, no KV handoff.

## Object layer check

The hybrid team with the rule centre ran through `CommandCentre`, envelopes and `TraceLog` on 12 test incidents with ollama:qwen2.5-coder:14b: top-1 92%, joint 50% (a smoke-sized subset, not a result). 56 envelopes per incident, 30,656 bytes on the wire, of which 2,487 bytes are the typed reports. The batch path's rule centre on the same reports gave the same ranking (True) and fault (True).
