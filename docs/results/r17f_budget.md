# R17 step 5b: how much tuning time a mission is worth

Exploratory (no bar): mission-time tuning with 25 to 400 generations on the same 40 + 40 held-out missions (step 5 used 100). One CPU core per mission; scores are found − 2 × wrong find, in points.

| Generations | Seconds (median) | 85% walking hint | Still person, hint "still" | − step-2 planner (hint) | − step-1 planner (still) |
|---|---|---|---|---|---|
| 25 | 10 | 31.3 | 59.9 | +2.0 ± 1.9 | -4.7 ± 1.9 |
| 50 | 19 | 33.1 | 61.4 | +3.9 ± 2.1 | -3.2 ± 2.0 |
| 100 | 30 | 34.1 | 63.1 | +4.8 ± 2.4 | -1.5 ± 1.9 |
| 200 | 69 | 34.5 | 64.1 | +5.2 ± 3.1 | -0.5 ± 1.5 |
| 400 | 130 | 35.2 | 65.7 | +5.9 ± 3.9 | +1.1 ± 1.3 |

References: step-2 planner 29.3 / 49.2; step-1 planner 18.4 / 64.6.

Reproduce: `python scripts/r17f_mission_time.py --generations <n> --from runs:r17-train-cond-* runs:r17b-train-cond-* runs:r17b-oracle`.
