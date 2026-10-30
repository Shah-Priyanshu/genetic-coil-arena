# R17 step 3b: a bigger planner reading the moving hint (32 hidden units, 6,000 generations)

**NOT YET** (all three bars needed).

| Bar | Difference (points, 95% CI over missions) | Needed | Met |
|---|---|---|---|
| (a) still people: v3 − step-1 planner (still person, hint still) | -18.4 ± 3.3 | ≥ −3 | no |
| (b) gain: v3 − v2 (hint 85% right) | -2.1 ± 1.7 | ≥ +3, CI > 0 | no |
| (c) walking kept: v3 − v2 on walking missions (hint 85% right) | -3.8 ± 2.6 | ≥ −3 | no |

Score (found − 2 × wrong find, points); 40 held-out missions × 2000 episodes per condition (23 with a walking person in the version 2 set).

| Condition | step-1 planner | planner v2 | planner v3 (hint) |
|---|---|---|---|
| hint 85% right | 18.4 | 29.3 | 27.2 |
| hint always right | 18.4 | 29.3 | 27.8 |
| hint always wrong | 18.4 | 29.3 | 21.9 |
| no hint | 18.4 | 29.3 | 26.2 |
| still person, hint still | 64.6 | 49.2 | 46.2 |

Reproduce: `python scripts/r17c_hint.py --from runs:r17-train-cond-* runs:r17b-train-cond-* runs:r17c-train-*`.
