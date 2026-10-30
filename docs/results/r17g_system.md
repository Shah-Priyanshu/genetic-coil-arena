# R17 step 6: the recommended search system on realistic missions

**PASS.** The recommended system (448 camera, false detections cut 10×, 3 frames per look, operator confirms, mission-time tuning) finds the person in 52.3% of episodes with 4.8% wrong finds (bar: ≥ 50% and ≤ 5%).

Step 3's 40 held-out missions (half with a walking person; GPS error, wind, battery vary; the moving hint is right 85% of the time) × 2000 episodes. Found / wrong finds, averaged over missions.

| Camera | False detections | Frames per look | OpenSwarm default | Step-2 planner | Mission-time tuning |
|---|---|---|---|---|---|
| K210 224 | ×1 | 3 | 5% / 94.6% | 4% / 65.5% | 10% / 6.6% |
| K210 224 | ×0.1 | 3 | 19% / 40.4% | 9% / 44.5% | 36% / 4.2% |
| 448 camera | ×1 | 3 | 2% / 97.9% | 4% / 68.5% | 16% / 13.4% |
| 448 camera | ×0.1 | 3 | 19% / 72.9% | 7% / 55.4% | 52% / 4.8% |
| 448 camera | ×0.1 | 1 | 16% / 46.4% | 9% / 42.0% | 35% / 4.4% |

Assumptions: a person is 0.34 m in R15b's size measure; the OV2640's ~60° lens on both cameras; the operator recognises a person in view 90% of the time per look and wrongly accepts an empty view 2% of the time; the ×0.1 filter cuts false detections and leaves recall unchanged.

Reproduce: `python scripts/r17g_system.py --from runs:r17b-train-cond-*`.
