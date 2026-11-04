# R23c: the K210 scout brain with a finer heatmap and size-aware calibration

**PASS.** Variant chosen on seed 0's val: **heat** (people found at float's val false-detection rate, interpolated between the int8 score steps):

| Variant | People found at float's val false rate | Threshold set on val | 32+ px people at that threshold |
|---|---|---|---|
| base | 33.5% | 0.522 | 48.7% |
| heat | 34.1% | 0.506 | 53.0% |
| heat_sized | 33.2% | 0.518 | 50.6% |

Float on val at 0.5: 36.1% of people, 0.215 false detections per tile.

Test-dev, the chosen variant with R23b's threshold rule and bars (people found at most 2 points below float's at 0.5, false detections per tile within 10%):

| Seed | int8 threshold | Recall float / int8 | False per tile float / int8 | 0-8 px | 8-16 px | 16-32 px | 32+ px | Passes |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.506 | 23.8% / 22.2% | 0.099 / 0.092 (-7%) | 10.6 / 10.2 | 29.0 / 27.8 | 45.1 / 40.9 | 62.3 / 53.4 | yes |
| 1 | 0.494 | 23.3% / 21.7% | 0.093 / 0.092 (-1%) | 10.3 / 10.0 | 28.4 / 27.1 | 44.5 / 40.3 | 61.0 / 50.8 | yes |

Reproduce: `python scripts/r23_k210.py select-c`, then `report-c`.
