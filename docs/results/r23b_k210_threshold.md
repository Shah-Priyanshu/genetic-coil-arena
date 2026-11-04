# R23b: the K210 scout brain with its own threshold

**NOT PASSED.** Per seed the compiled model's threshold is set on VisDrone val so that its false detections per tile do not exceed the float brain's at 0.5; test-dev is then compared at those thresholds. Bars: people found (all sizes) at most 2 points below float's, false detections per tile within 10%.

| Seed | int8 threshold | Recall float / int8 | False per tile float / int8 | 0-8 px | 8-16 px | 16-32 px | 32+ px | Passes |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.522 | 23.8% / 20.3% | 0.099 / 0.073 (-27%) | 10.6 / 9.3 | 29.0 / 25.7 | 45.1 / 36.9 | 62.3 / 50.1 | no |
| 1 | 0.500 | 23.3% / 20.7% | 0.093 / 0.080 (-13%) | 10.3 / 9.5 | 28.4 / 26.2 | 44.5 / 38.1 | 61.0 / 49.2 | no |

Recall by person size (px in the 224 tile), float at 0.5 / int8 at its threshold. Reproduce: `python scripts/r23_k210.py report-b`.
