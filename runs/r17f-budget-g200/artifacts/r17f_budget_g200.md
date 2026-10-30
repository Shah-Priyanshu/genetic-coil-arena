# R17 step 5: plan at mission time

**GOOD SIGN.**

- (a) mission-time tuning − step-2 planner, hint 85% right: +5.2 ± 3.1 points (bar ≥ +5, CI > 0): met.
- (b) mission-time tuning − step-1 planner, still person with the hint "still": -0.5 ± 1.5 points (bar ≥ -3): met.

Tuning took 69 s per mission on one CPU core (median; max 165 s); the step-2 planner's own plan was the better start in 39% of missions.

Score (found − 2 × wrong find, points), 40 held-out missions × 2000 episodes:

| System | hint 85% right | still person, hint still |
|---|---|---|
| mission-time tuning | 34.5 | 64.1 |
| step-1 planner | 18.4 | 64.6 |
| step-2 planner | 29.3 | 49.2 |
| tuned knowing the person | 38.7 | — |

Reproduce: `python scripts/r17f_mission_time.py --from runs:r17-train-cond-* runs:r17b-train-cond-* runs:r17b-oracle`.
