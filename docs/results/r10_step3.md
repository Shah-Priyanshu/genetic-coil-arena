# R10 step 3: trained fault-type specialists (goal G4c)

Step 3 of R10 on RCAEval RE2. The rule centre picks the service by the metric specialists' anomaly score (as step 2) and takes the fault type from a *trained* specialist's typed report for that service: a numpy classifier on per-kind metric features (CPU) and a LoRA adapter on Qwen3-1.7B reading the service's own log view (GPU), alone or fused. Training data: Train Ticket (all repetitions) and Sock Shop / Online Boutique repetition 1; the test split (SS/OB repetitions 2 and 3, 120 incidents) is never fitted or selected on.

## Verdict

**G4c: PASS.** Pass: a trained team beats BARO on joint accuracy by at least 5 points (paired-bootstrap 95% CI excludes 0, Holm-adjusted p < 0.05 over the trained teams) with top-1 non-inferior (CI lower bound above -3 points). Kill: every trained team is within the CI of BARO on joint accuracy.

| Trained team | Team joint | BARO joint | Difference [95% CI] | p (Holm) | Top-1 difference [95% CI] | Verdict |
|---|---|---|---|---|---|---|
| team_trained_metric | 68.3 | 45.0 | +23.3 [+15.0, +31.7] | 1.57e-06 | +0.0 [+0.0, +0.0] | **PASS** |
| team_trained_log | 18.3 | 45.0 | -26.7 [-35.8, -18.3] | 6.27e-07 | +0.0 [+0.0, +0.0] | **KILL** |
| team_trained_both | 69.2 | 45.0 | +24.2 [+15.8, +32.5] | 8.92e-07 | +0.0 [+0.0, +0.0] | **PASS** |

All trained teams rank services by the same metric score as BARO; check: the top service equals BARO's on 100.0% (team_trained_metric), 100.0% (team_trained_log), 100.0% (team_trained_both), 100.0% (team_trained_metric_ttonly) of the test incidents, so top-1 is BARO's by construction and the whole difference is the fault type.

## Analysis

<!-- analysis:start -->
- **The trained metric specialist passes G4c.** Its typed report lifts joint accuracy from BARO's 45.0 to 68.3: +23.3 [+15.0, +31.7], Holm p 9e-6.
  - Every incident's top service matches BARO's, so the whole gain is fault type: 48.3 → 72.5.
  - It comes from the faults that were the bottleneck in step 2. Joint accuracy rose from 20 to 80 for socket, 0 to 40 for loss and 5 to 35 for disk; cpu, delay and mem held at 75–90.
- **Most of the gain carries over to systems the classifier never saw.** A test incident repeats a training case (same service, same fault), so part of the gain could be memorisation.
  - The specialist trained on Train Ticket only never saw Sock Shop or Online Boutique. It still beats BARO by +18.3 [+10.8, +25.8].
  - Adding the SS/OB repetition-1 data adds +5.0 [+0.8, +10.0] more (Holm p 0.068, not significant). Almost all of that is loss, where fault accuracy rose from 20 to 60.
  - Leave-one-system-out accuracy on the training data (56.7–70.0, against 40.0–43.3 for step 2's heuristic) agrees with the test numbers.
- **The centre uses the report.**
  - Dropping the root service's fault report sends the centre back to the step-2 heuristic, at exactly BARO's 45.0.
  - Swapping in another incident's fault report drops joint accuracy to 8.3.
- **The LoRA log specialist learned nothing that carries over.**
  - It fit its training views (loss 0.91, against ln 6 = 1.79 for a uniform guess). Its held-out fault accuracy was 6.7–16.7%, against 16.7% chance.
  - It scored 19.2% on the test split even though the final adapter had trained on the same cases.
  - Fusing it with the metric specialist (rule `geo_w50`, chosen on out-of-fold training predictions) adds +0.8 [−1.7, +4.2].
  - Three steps now agree: RE2's templated logs carry next to no root-cause or fault-type signal. Here the signal is in the metrics, and a 2 KB typed report carries it.
- **What G4c does and does not show.**
  - It shows the specialist pattern working end to end: a trained specialist reads its own service's data and sends a schema-validated report with calibrated fault probabilities, and a simple centre combines the reports. The result beats the best single system.
  - The specialist here is a 59-feature softmax regression, not an LLM, so this says nothing yet for LLM specialists. That question moves to step 4 on RCAEval RE3, whose code-level faults should show up in the logs (G4d).
- **Caveats.**
  - One dataset, 120 test incidents (20 per fault type), a rule centre.
  - The trained team has not been run through `CommandCentre` and envelopes; the step-2 `RuntimeTeam` shows that path works for the untrained team.
  - The fault labels are RCAEval's injected faults.
<!-- analysis:end -->

## Results (test split)

Percent per incident with 95% bootstrap CI. Joint = root-cause service and fault type both right. Chance for top-1 is about 7.5, for fault type 16.7, for joint about 1.3. Bytes are the typed reports' compact JSON per incident (the trained reports carry six fault probabilities each).

| System | n | Top-1 | Top-3 | Fault type | Joint | Bytes / incident |
|---|---|---|---|---|---|---|
| baro_metrics | 120 | 88.3 [82.5, 93.3] | 99.2 [97.5, 100.0] | 48.3 [39.2, 57.5] | 45.0 [35.8, 53.3] | 0 |
| team_metric_rule | 120 | 88.3 [82.5, 93.3] | 99.2 [97.5, 100.0] | 48.3 [39.2, 57.5] | 45.0 [35.8, 53.3] | 1,122 |
| team_trained_metric | 120 | 88.3 [82.5, 93.3] | 99.2 [97.5, 100.0] | 72.5 [64.2, 80.0] | 68.3 [60.0, 76.7] | 2,097 |
| team_trained_log | 120 | 88.3 [82.5, 93.3] | 99.2 [97.5, 100.0] | 19.2 [12.5, 26.7] | 18.3 [11.7, 25.0] | 3,716 |
| team_trained_both | 120 | 88.3 [82.5, 93.3] | 99.2 [97.5, 100.0] | 72.5 [64.2, 80.0] | 69.2 [60.8, 77.5] | 3,667 |
| team_trained_metric_ttonly | 120 | 88.3 [82.5, 93.3] | 99.2 [97.5, 100.0] | 65.8 [57.5, 74.2] | 63.3 [55.0, 71.7] | 2,097 |

### Paired comparisons of joint accuracy

Difference a - b in points on shared test incidents, paired bootstrap 95% CI, Wilcoxon p, Holm over this table.

| a | b | n | Difference | 95% CI | p | p (Holm) |
|---|---|---|---|---|---|---|
| team_trained_metric | baro_metrics | 120 | +23.3 | [+15.0, +31.7] | 1.57e-06 | 9.43e-06 |
| team_trained_metric | team_metric_rule | 120 | +23.3 | [+15.0, +31.7] | 1.57e-06 | 9.43e-06 |
| team_trained_log | baro_metrics | 120 | -26.7 | [-35.8, -18.3] | 2.09e-07 | 2.09e-06 |
| team_trained_log | team_metric_rule | 120 | -26.7 | [-35.8, -18.3] | 2.09e-07 | 2.09e-06 |
| team_trained_both | baro_metrics | 120 | +24.2 | [+15.8, +32.5] | 4.46e-07 | 3.57e-06 |
| team_trained_both | team_metric_rule | 120 | +24.2 | [+15.8, +32.5] | 4.46e-07 | 3.57e-06 |
| team_trained_metric_ttonly | baro_metrics | 120 | +18.3 | [+10.8, +25.8] | 1.6e-05 | 6.4e-05 |
| team_trained_metric_ttonly | team_metric_rule | 120 | +18.3 | [+10.8, +25.8] | 1.6e-05 | 6.4e-05 |
| team_trained_both | team_trained_metric | 120 | +0.8 | [-1.7, +4.2] | 0.564 | 0.564 |
| team_trained_both | team_trained_log | 120 | +50.8 | [+41.7, +60.0] | 1.53e-14 | 1.68e-13 |
| team_trained_metric | team_trained_metric_ttonly | 120 | +5.0 | [+0.8, +10.0] | 0.0339 | 0.0678 |

### Joint accuracy by fault type and system

| Group | n | baro_metrics | team_metric_rule | team_trained_metric | team_trained_log | team_trained_both | team_trained_metric_ttonly |
|---|---|---|---|---|---|---|---|
| fault cpu | 20 | 75.0 | 75.0 | 80.0 | 55.0 | 75.0 | 75.0 |
| fault delay | 20 | 85.0 | 85.0 | 85.0 | 15.0 | 85.0 | 85.0 |
| fault disk | 20 | 5.0 | 5.0 | 35.0 | 0.0 | 35.0 | 35.0 |
| fault loss | 20 | 0.0 | 0.0 | 40.0 | 5.0 | 45.0 | 10.0 |
| fault mem | 20 | 85.0 | 85.0 | 90.0 | 35.0 | 90.0 | 90.0 |
| fault socket | 20 | 20.0 | 20.0 | 80.0 | 0.0 | 85.0 | 85.0 |
| system OB | 60 | 45.0 | 45.0 | 61.7 | 23.3 | 61.7 | 56.7 |
| system SS | 60 | 45.0 | 45.0 | 75.0 | 13.3 | 76.7 | 70.0 |

### Fault-type accuracy by fault type and system (service right or wrong)

| Group | n | baro_metrics | team_metric_rule | team_trained_metric | team_trained_log | team_trained_both | team_trained_metric_ttonly |
|---|---|---|---|---|---|---|---|
| fault cpu | 20 | 75.0 | 75.0 | 80.0 | 60.0 | 75.0 | 75.0 |
| fault delay | 20 | 100.0 | 100.0 | 90.0 | 15.0 | 95.0 | 90.0 |
| fault disk | 20 | 5.0 | 5.0 | 35.0 | 0.0 | 35.0 | 35.0 |
| fault loss | 20 | 5.0 | 5.0 | 60.0 | 5.0 | 55.0 | 20.0 |
| fault mem | 20 | 85.0 | 85.0 | 90.0 | 35.0 | 90.0 | 90.0 |
| fault socket | 20 | 20.0 | 20.0 | 80.0 | 0.0 | 85.0 | 85.0 |
| system OB | 60 | 48.3 | 48.3 | 66.7 | 25.0 | 66.7 | 58.3 |
| system SS | 60 | 48.3 | 48.3 | 78.3 | 13.3 | 78.3 | 73.3 |

### Message ablations

`drop_*` / `shuffle_*` empty or swap both reports of one service for another incident's (root = the true root-cause service, other = a random other one). `*_fault_root` touches only the root service's fault reports (scores and onset stay), so the centre falls back to the metric-kind heuristic (drop) or reads another incident's fault report (shuffle).

| Team | Ablation | n | Top-1 intact | Top-1 ablated | Change [95% CI] | Joint intact | Joint ablated | Change [95% CI] |
|---|---|---|---|---|---|---|---|---|
| team_trained_metric | drop_fault_root | 120 | 88.3 | 88.3 | +0.0 [+0.0, +0.0] | 68.3 | 45.0 | -23.3 [-31.7, -15.0] |
| team_trained_metric | drop_other | 120 | 88.3 | 91.7 | +3.3 [+0.8, +6.7] | 68.3 | 70.0 | +1.7 [+0.0, +4.2] |
| team_trained_metric | drop_root | 120 | 88.3 | 0.0 | -88.3 [-94.2, -82.5] | 68.3 | 0.0 | -68.3 [-76.7, -60.0] |
| team_trained_metric | shuffle_fault_root | 120 | 88.3 | 88.3 | +0.0 [+0.0, +0.0] | 68.3 | 8.3 | -60.0 [-69.2, -50.0] |
| team_trained_metric | shuffle_other | 120 | 88.3 | 90.0 | +1.7 [-1.7, +5.0] | 68.3 | 68.3 | +0.0 [-2.5, +2.5] |
| team_trained_metric | shuffle_root | 120 | 88.3 | 29.2 | -59.2 [-68.3, -50.0] | 68.3 | 2.5 | -65.8 [-74.2, -56.7] |
| team_trained_log | drop_fault_root | 120 | 88.3 | 88.3 | +0.0 [+0.0, +0.0] | 18.3 | 45.0 | +26.7 [+18.3, +35.8] |
| team_trained_log | drop_other | 120 | 88.3 | 91.7 | +3.3 [+0.8, +6.7] | 18.3 | 18.3 | +0.0 [+0.0, +0.0] |
| team_trained_log | drop_root | 120 | 88.3 | 0.0 | -88.3 [-94.2, -82.5] | 18.3 | 0.0 | -18.3 [-25.0, -11.7] |
| team_trained_log | shuffle_fault_root | 120 | 88.3 | 88.3 | +0.0 [+0.0, +0.0] | 18.3 | 15.8 | -2.5 [-9.2, +5.0] |
| team_trained_log | shuffle_other | 120 | 88.3 | 90.0 | +1.7 [-1.7, +5.0] | 18.3 | 18.3 | +0.0 [+0.0, +0.0] |
| team_trained_log | shuffle_root | 120 | 88.3 | 29.2 | -59.2 [-68.3, -50.0] | 18.3 | 6.7 | -11.7 [-19.2, -3.3] |
| team_trained_both | drop_fault_root | 120 | 88.3 | 88.3 | +0.0 [+0.0, +0.0] | 69.2 | 45.0 | -24.2 [-32.5, -15.8] |
| team_trained_both | drop_other | 120 | 88.3 | 91.7 | +3.3 [+0.8, +6.7] | 69.2 | 70.8 | +1.7 [+0.0, +4.2] |
| team_trained_both | drop_root | 120 | 88.3 | 0.0 | -88.3 [-94.2, -82.5] | 69.2 | 0.0 | -69.2 [-77.5, -60.8] |
| team_trained_both | shuffle_fault_root | 120 | 88.3 | 88.3 | +0.0 [+0.0, +0.0] | 69.2 | 7.5 | -61.7 [-70.8, -52.5] |
| team_trained_both | shuffle_other | 120 | 88.3 | 90.0 | +1.7 [-1.7, +5.0] | 69.2 | 69.2 | +0.0 [-2.5, +2.5] |
| team_trained_both | shuffle_root | 120 | 88.3 | 29.2 | -59.2 [-68.3, -50.0] | 69.2 | 3.3 | -65.8 [-74.2, -56.7] |
| team_trained_metric_ttonly | drop_fault_root | 120 | 88.3 | 88.3 | +0.0 [+0.0, +0.0] | 63.3 | 45.0 | -18.3 [-26.7, -10.8] |
| team_trained_metric_ttonly | drop_other | 120 | 88.3 | 91.7 | +3.3 [+0.8, +6.7] | 63.3 | 63.3 | +0.0 [+0.0, +0.0] |
| team_trained_metric_ttonly | drop_root | 120 | 88.3 | 0.0 | -88.3 [-94.2, -82.5] | 63.3 | 0.0 | -63.3 [-71.7, -55.0] |
| team_trained_metric_ttonly | shuffle_fault_root | 120 | 88.3 | 88.3 | +0.0 [+0.0, +0.0] | 63.3 | 8.3 | -55.0 [-65.0, -44.2] |
| team_trained_metric_ttonly | shuffle_other | 120 | 88.3 | 90.0 | +1.7 [-1.7, +5.0] | 63.3 | 63.3 | +0.0 [+0.0, +0.0] |
| team_trained_metric_ttonly | shuffle_root | 120 | 88.3 | 29.2 | -59.2 [-68.3, -50.0] | 63.3 | 3.3 | -60.0 [-69.2, -50.8] |

## Training and cross-validation (training data only)

Metric fault specialist: 150 training rows (root-cause services); leave-one-system-out accuracy of the chosen configuration against step 2's metric-kind heuristic on the same rows.

| System held out | Trained specialist | Metric-kind heuristic |
|---|---|---|
| OB | 70.0 | 43.3 |
| SS | 66.7 | 40.0 |
| TT | 56.7 | 43.3 |

Chosen configuration: `{'hidden': 0, 'l2': 1.0, 'epochs': 400}`. Per fault type (all training systems pooled):

| Fault | n | Specialist (out-of-fold) | Heuristic |
|---|---|---|---|
| cpu | 25 | 76.0 | 60.0 |
| mem | 25 | 36.0 | 32.0 |
| disk | 25 | 72.0 | 60.0 |
| delay | 25 | 80.0 | 100.0 |
| loss | 25 | 24.0 | 0.0 |
| socket | 25 | 80.0 | 4.0 |

| Configuration | OB | SS | TT | Mean |
|---|---|---|---|---|
| {'hidden': 0, 'l2': 0.01, 'epochs': 400} | 70.0 | 66.7 | 43.3 | 60.0 |
| {'hidden': 0, 'l2': 0.1, 'epochs': 400} | 70.0 | 70.0 | 46.7 | 62.2 |
| {'hidden': 0, 'l2': 1.0, 'epochs': 400} | 70.0 | 66.7 | 56.7 | 64.4 |
| {'hidden': 16, 'l2': 0.01, 'epochs': 300, 'n_models': 5} | 73.3 | 70.0 | 43.3 | 62.2 |
| {'hidden': 16, 'l2': 0.1, 'epochs': 300, 'n_models': 5} | 76.7 | 66.7 | 50.0 | 64.4 |
| {'hidden': 32, 'l2': 0.1, 'epochs': 300, 'n_models': 5} | 73.3 | 66.7 | 50.0 | 63.3 |
| {'hidden': 32, 'l2': 1.0, 'epochs': 300, 'n_models': 5} | 30.0 | 33.3 | 34.4 | 32.6 |

Fusion rule, chosen on 150 out-of-fold root-service reports (`{'OB': 30, 'SS': 30, 'TT': 90}`): **geo_w50**.

| Rule | OB | SS | TT | mean |
|---|---|---|---|---|
| metric | 70.0 | 66.7 | 56.7 | 64.4 |
| log | 16.7 | 6.7 | 14.4 | 12.6 |
| mean_w25 | 73.3 | 66.7 | 57.8 | 65.9 |
| mean_w50 | 73.3 | 66.7 | 57.8 | 65.9 |
| mean_w75 | 66.7 | 63.3 | 56.7 | 62.2 |
| geo_w25 | 73.3 | 66.7 | 57.8 | 65.9 |
| geo_w50 | 73.3 | 70.0 | 57.8 | 67.0 |
| geo_w75 | 66.7 | 63.3 | 55.6 | 61.9 |
| max_conf | 70.0 | 66.7 | 56.7 | 64.4 |
| stack | 46.7 | 43.3 | 42.2 | 44.1 |

Log fault specialist (LoRA): one adapter per fold plus the final one.

| Fold | Held out | Fit incidents | Examples | Steps | Final loss | Train (s) | Peak VRAM (MiB) | Held-out fault accuracy |
|---|---|---|---|---|---|---|---|---|
| final | - | 150 | 450 | 225 | 0.913 | 481 | 5604 | - |
| ob | OB | 120 | 360 | 180 | 0.900 | 375 | 5603 | 16.7 |
| ss | SS | 120 | 360 | 180 | 0.904 | 379 | 5604 | 6.7 |
| tt | TT | 60 | 180 | 90 | 0.911 | 194 | 5600 | 14.4 |

## Setup

- **Splits:** RCAEval RE2. Training = Train Ticket (all repetitions) + SS/OB repetition 1; test = SS/OB repetitions 2 and 3. A test incident repeats a dev case (same service and fault), so models fitted on SS/OB-dev may partly memorise cases; leave-one-system-out cross-validation and the TT-only diagnostic measure cross-system transfer instead. `scripts/r10_step3.py data` checks the split and `tests/test_log_triage3.py` pins it.
- **Metric fault specialist:** per service, features by metric kind from its own metrics (z, direction, relative change, onset per kind; share of the strongest kind; count of kinds beyond z 5 and 20). A class-balanced numpy softmax regression or small MLP ensemble, chosen by leave-one-system-out accuracy over TT / SS-dev / OB-dev. Typed `TRAINED_METRIC_SCHEMA` report: step 2's fields plus `fault` and six `fault_probs`.
- **Log fault specialist:** Qwen3-1.7B (bfloat16) with one LoRA adapter (rank 8, all linear layers), trained on the root service's step-2 log view (-120 s to +300 s around the reported onset, 1,500 tokens, three onset jitters as augmentation) with the fault letter as the target. Scored by closed-set letter log-probabilities. Typed `TRAINED_LOG_SCHEMA` report.
- **Centre:** rule. Ranking by metric score; fault = argmax of the specialist's distribution for the top service (metric, log) or of the fused distribution (fusion rule chosen on out-of-fold predictions of the training data only). Missing or empty reports fall back to step 2's metric-kind heuristic.
- **Not done:** no CommandCentre/envelope run of the trained team, no LLM centre, no trained service ranking.
