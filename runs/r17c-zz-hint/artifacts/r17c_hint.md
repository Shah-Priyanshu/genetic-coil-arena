# R17 step 3: tell the leader whether the person may be moving

**NOT YET** (all three bars needed).

| Bar | Difference (points, 95% CI over missions) | Needed | Met |
|---|---|---|---|
| (a) still people: v3 − step-1 planner (still person, hint still) | -15.6 ± 2.9 | ≥ −3 | no |
| (b) gain: v3 − v2 (hint 85% right) | -0.9 ± 1.8 | ≥ +3, CI > 0 | no |
| (c) walking kept: v3 − v2 on walking missions (hint 85% right) | -2.3 ± 2.8 | ≥ −3 | yes |

Score (found − 2 × wrong find, points); 40 held-out missions × 2000 episodes per condition (23 with a walking person in the version 2 set).

| Condition | step-1 planner | planner v2 | planner v3 (hint) |
|---|---|---|---|
| hint 85% right | 18.4 | 29.3 | 28.4 |
| hint always right | 18.4 | 29.3 | 29.4 |
| hint always wrong | 18.4 | 29.3 | 22.3 |
| no hint | 18.4 | 29.3 | 26.8 |
| still person, hint still | 64.6 | 49.2 | 49.0 |

Reproduce: `python scripts/r17c_hint.py --from runs:r17-train-cond-* runs:r17b-train-cond-* runs:r17c-train-*`.
