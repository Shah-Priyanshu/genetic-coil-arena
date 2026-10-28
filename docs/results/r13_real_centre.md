# R13: the frozen G4c team on fresh incidents, and a learned centre (G11a, G11b)

Generated 2026-10-28 01:06 by `scripts/r13_real_centre.py report`. RCAEval RE1 (metrics only, no logs; resource faults cpu, mem, disk, delay, loss), never used by an earlier goal. Differences are in percentage points with paired bootstrap 95% CIs (10,000 resamples over incidents).

Built locally from the `r13-replicate`, `r13-train` and `r13-test` job artifacts (pinned 9126dd6a), the same inputs as the runner's `r13-zz-aggregate`.

## G11a: does G4c replicate on RE1? **partial**

Frozen fault specialist SHA-256 `953524c1ce6905fc`, team state hash `f83d49a1fea235a5` (asserted unchanged after the run). 375 incidents. Pass: pooled joint gain >= 5 points over BARO with CI above 0, and the same in each system. Kill: pooled CI includes 0 or lies below.

| Metric | System | Team | BARO | Difference [95% CI] |
|---|---|---|---|---|
| Joint | pooled | 46.7 | 37.3 | +9.3 [+5.3, +13.3] |
| Joint | SS | 55.2 | 33.6 | +21.6 [+14.4, +28.8] |
| Joint | OB | 52.8 | 54.4 | -1.6 [-9.6, +6.4] |
| Joint | TT | 32.0 | 24.0 | +8.0 [+3.2, +12.8] |
| Root top-1 | pooled | 82.7 | 82.7 | +0.0 [+0.0, +0.0] |
| Root top-1 | SS | 89.6 | 89.6 | +0.0 [+0.0, +0.0] |
| Root top-1 | OB | 86.4 | 86.4 | +0.0 [+0.0, +0.0] |
| Root top-1 | TT | 72.0 | 72.0 | +0.0 [+0.0, +0.0] |
| Fault type | pooled | 58.9 | 44.5 | +14.4 [+10.1, +18.7] |
| Fault type | SS | 64.0 | 37.6 | +26.4 [+18.4, +34.4] |
| Fault type | OB | 65.6 | 59.2 | +6.4 [-2.4, +15.2] |
| Fault type | TT | 47.2 | 36.8 | +10.4 [+5.6, +16.0] |

Team top service equals BARO's in 375 of 375 incidents (by construction).

### Joint accuracy per fault type

| Fault | n | Team | BARO |
|---|---|---|---|
| cpu | 75 | 100.0 | 78.7 |
| mem | 75 | 38.7 | 44.0 |
| disk | 75 | 0.0 | 0.0 |
| delay | 75 | 64.0 | 64.0 |
| loss | 75 | 30.7 | 0.0 |

### Fault confusion, team (rows: true, columns: predicted)

| true | cpu | mem | disk | delay | loss | socket |
|---|---|---|---|---|---|---|
| cpu | 75 | 0 | 0 | 0 | 0 | 0 |
| mem | 16 | 29 | 20 | 7 | 3 | 0 |
| disk | 71 | 0 | 1 | 0 | 3 | 0 |
| delay | 1 | 0 | 0 | 74 | 0 | 0 |
| loss | 3 | 0 | 0 | 30 | 42 | 0 |

### Fault confusion, baro (rows: true, columns: predicted)

| true | cpu | mem | disk | delay | loss | socket |
|---|---|---|---|---|---|---|
| cpu | 59 | 0 | 0 | 16 | 0 | 0 |
| mem | 17 | 33 | 0 | 19 | 6 | 0 |
| disk | 55 | 0 | 0 | 19 | 1 | 0 |
| delay | 1 | 0 | 0 | 74 | 0 | 0 |
| loss | 1 | 0 | 0 | 73 | 1 | 0 |

### RE1 data

| System | Incidents | Cases found | Parse failures | Metric kinds (incidents having each) |
|---|---|---|---|---|
| RE1-SS | 125 | 125 | 0 | cpu 125, error 125, latency-50 125, latency-90 125, mem 125, workload 125 |
| RE1-OB | 125 | 125 | 0 | cpu 125, error 124, latency-50 125, latency-90 75, mem 125, workload 124 |
| RE1-TT | 125 | 125 | 0 | cpu 125, error 125, latency-50 125, latency-90 125, mem 125, workload 125 |

RE1 has no logs. Renamed kinds (Online Boutique): `load` as `workload`, `latency` as `latency-50`.

## G11b: LOSO cross-validation on RE1

375 RE1 incidents. Mean held-out top-1 over the three left-out systems; selection rule: highest mean, ties to the compact set then the larger L2.

| # | Features | L2 | SS held out | OB held out | TT held out | Mean |
|---|---|---|---|---|---|---|
| 0 | compact | 0.01 | 89.6 | 86.4 | 72.0 | 82.7 |
| 1 | compact | 0.1 | 89.6 | 86.4 | 72.0 | 82.7 |
| 2 | compact | 1.0 | 89.6 | 86.4 | 72.0 | 82.7 |
| 3 | all | 0.01 | 89.6 | 82.4 | 80.8 | 84.3 |
| 4 * | all | 0.1 | 89.6 | 86.4 | 77.6 | 84.5 |
| 5 | all | 1.0 | 88.8 | 87.2 | 75.2 | 83.7 |
| BARO | | | 89.6 | 86.4 | 72.0 | |

Chosen config (*), learned minus BARO top-1 on the held-out systems: pooled +1.9 [-0.5, +4.3]; SS +0.0 [-5.6, +5.6]; OB +0.0 [-2.4, +2.4]; TT +5.6 [+1.6, +9.6].

## G11b: the ranker on RE2 **kill**

Ranker `3da0eeb92c5a83ab` (all, L2 0.1), trained on RE1 only, tested once on 210 RE2 incidents (SS 60, OB 60, TT 90). These RE2 splits were used by earlier goals (G4c to G10b), but never to train or select this ranker. The frozen fault specialist was trained on RE2 Train Ticket, so its probabilities are optimistic there. Pass: pooled top-1 gain >= 3 points over the rule centre (BARO ranking) with CI above 0, and each system's CI lower bound above -5. Kill: pooled CI includes 0 or lies below.

| System | Learned | Rule centre | Difference [95% CI] |
|---|---|---|---|
| pooled | 86.7 | 84.3 | +2.4 [-1.0, +5.7] |
| SS | 98.3 | 93.3 | +5.0 [+0.0, +11.7] |
| OB | 91.7 | 83.3 | +8.3 [+1.7, +15.0] |
| TT | 75.6 | 78.9 | -3.3 [-8.9, +1.1] |

Not a criterion: joint accuracy with the specialist's fault for the chosen service, pooled: learned 71.9, rule 71.0, +1.0 [-1.9, +3.8].

## Metric-kind coverage, RE1 against RE2

Share of incidents whose root service reports each kind.

| Group | n | cpu | mem | socket | diskio | workload | latency-50 | latency-90 | error |
|---|---|---|---|---|---|---|---|---|---|
| RE1 SS | 125 | 100.0 | 100.0 | 0.0 | 0.0 | 100.0 | 100.0 | 100.0 | 52.0 |
| RE1 OB | 125 | 100.0 | 100.0 | 0.0 | 0.0 | 99.2 | 100.0 | 60.0 | 8.8 |
| RE1 TT | 125 | 100.0 | 100.0 | 0.0 | 0.0 | 100.0 | 100.0 | 100.0 | 28.0 |
| RE2 SS | 60 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 20.0 |
| RE2 OB | 60 | 100.0 | 100.0 | 100.0 | 71.7 | 100.0 | 100.0 | 100.0 | 0.0 |
| RE2 TT | 90 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 32.2 |

<!-- analysis:start -->
**G11a is "partial": the trained fault specialist's advantage replicates on fresh incidents, at about 40% of its RE2 size, and not on Online Boutique.**

| RE1 (375 fresh incidents) | Team − BARO, joint accuracy |
|---|---|
| Pooled | +9.3 [+5.3, +13.3] (46.7 against 37.3) |
| Sock Shop | +21.6 |
| Train Ticket | +8.0 |
| Online Boutique | −1.6 [−9.6, +6.4] |

The pooled pass criterion is met. The pre-registered per-system condition is not, so the verdict is "partial". Root top-1 is identical (82.7) by construction: the rule centre ranks services by BARO. The whole difference is the fault type, 58.9% against 44.5%.

**The gap to RE2 (+23 on G4c's test) is mostly missing metrics.** RE1 records no disk-I/O and no socket metrics: 0% of root services have either, against 72 to 100% on RE2.
- **Disk:** without disk I/O, the specialist calls 71 of 75 disk faults "cpu". Disk joint accuracy is 0 for both systems.
- **Memory:** 29 of 75 right, with the rest spread over cpu, disk and delay. The specialist (38.7) is slightly below BARO's fault rule here (44.0).
- **Where it holds:**

  | Fault | Specialist | BARO |
  |---|---|---|
  | CPU | 75/75 | 59/75 |
  | Delay | 74/75 | 74/75 |
  | Loss | 42/75 | 1/75 |

  Loss is where BARO's rule calls almost every loss fault "delay".
- **Online Boutique** has the thinnest RE1 coverage: latency-90 in 60% of root services, error rates in 9%. It is the one system where the specialist does not beat BARO.

The specialist was frozen exactly as trained for G4c, with its state hash checked before and after. Nothing was refit for RE1.

**What G11a says.** A typed report carrying six fault probabilities from a small trained classifier keeps a significant advantage on 375 incidents it never saw, collected separately from its training data. The size of that advantage depends on which metric kinds the deployment records. The specialist's features are "by metric kind, never by name". That design transfers across systems and collections, but it cannot recover a fault whose defining metric is absent. Disk without disk I/O looks like CPU.

**G11b is a kill: a learned centre trained on RE1 does not beat the rule centre on RE2.**

| RE2 (210 incidents) | Learned − rule centre, root top-1 |
|---|---|
| Pooled | +2.4 [−1.0, +5.7] (86.7 against 84.3) |
| Sock Shop | +5.0 |
| Online Boutique | +8.3 |
| Train Ticket | −3.3 |

The per-system pattern is the reverse of its RE1 leave-one-system-out check. There Train Ticket was the only system it improved (+5.6), while Sock Shop and Online Boutique were unchanged. The pooled RE1 gain was +1.9 [−0.5, +4.3], already below the 3-point bar. So:
- with 375 training incidents and a few report features, the logistic ranker's gains are small and do not keep their sign across collections;
- the rule "take the service with the strongest metric anomaly" remains as good as anything learned here.

Joint accuracy, not a criterion, barely moves: +1.0 [−1.9, +3.8].

The test splits were not fresh in general: RE2 Sock Shop / Online Boutique and RE2 Train Ticket were used by earlier goals, though never to train or select this ranker. The frozen fault specialist was trained on RE2 Train Ticket, so its probabilities are optimistic on that part of the test, which, if anything, favours the learned centre there.

**Taken with G9 and G9b,** the real-incident results now point the same way. The parts that earn their place are small trained specialists sending typed reports (G4c, replicated here) and code (BARO, the novelty count). The centre that combines them is best kept as a simple rule:
- a hand gate (G9) costs the old class;
- a learned ranker (G11b) does not gain reliably.

Learned centres have only worked here on the squad's simulated goals (G7), where training data is unlimited.
<!-- analysis:end -->
