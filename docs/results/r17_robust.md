# R17 step 1b: does the mission-reading planner survive a worse world?

**Robust: FAIL.** The conditioned planner beats the blind plan by ≥ +3 points with the CI above 0 in all but: person walks 1.2 m/s, GPS error 3 m.

40 held-out missions × 2000 episodes per cell; score = found − 2 × wrong find, in points; gain = conditioned − blind, paired by mission (95% CI). Found / wrong find in brackets.

| Condition | OpenSwarm default | Hand two-tier | Blind (16) | Conditioned (16) | Gain |
|---|---|---|---|---|---|
| nominal | -69.8 (34% / 51.9%) | 38.2 (47% / 4.2%) | 46.6 (54% / 3.6%) | 64.6 (69% / 2.2%) | +18.0 ± 6.1 |
| person walks 0.5 m/s | -76.7 (29% / 52.7%) | -7.6 (3% / 5.4%) | -6.8 (3% / 4.9%) | -1.2 (6% / 3.8%) | +5.6 ± 4.1 |
| person walks 1.2 m/s | -74.5 (30% / 52.1%) | -9.9 (1% / 5.4%) | -8.8 (1% / 5.0%) | -5.7 (2% / 4.0%) | +3.1 ± 4.2 |
| GPS error 3 m | -100.7 (24% / 62.2%) | -7.3 (14% / 10.7%) | -6.0 (17% / 11.3%) | -3.7 (29% / 16.2%) | +2.2 ± 3.9 |
| headwind (3 m/s cruise) | -80.0 (30% / 55.2%) | 34.2 (43% / 4.4%) | 37.0 (44% / 3.6%) | 54.3 (59% / 2.4%) | +17.4 ± 5.6 |
| cold battery (240 s) | -68.0 (32% / 50.0%) | 34.5 (42% / 3.7%) | 39.2 (45% / 3.1%) | 56.2 (60% / 1.9%) | +17.0 ± 5.3 |
| areas x1.6 * | -107.3 (16% / 61.8%) | 15.7 (26% / 5.1%) | 17.0 (26% / 4.3%) | 31.1 (36% / 2.3%) | +14.0 ± 5.6 |
| 10 drones * | -75.9 (36% / 55.9%) | 45.9 (56% / 5.2%) | 60.5 (70% / 4.8%) | 74.1 (79% / 2.5%) | +13.6 ± 5.5 |

\* the planner's inputs (area, drones) leave the training range.

Reproduce: `python scripts/r17_robust.py --from runs:r17-train-*`.
