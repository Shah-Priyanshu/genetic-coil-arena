# R25b: the 448-px camera with a confirmer trained on its own candidates

**KILL.** Pooled over seeds, the 448 chain passes 1.32x [1.21, 1.44] the 224 chain's labelled false alarms per test-dev frame (bars: pooled <= 0.67x with the upper bound below 1, each seed <= 0.8x, people found within 1 point; KILL at >= 0.9x).

VisDrone2019-DET test-dev, 1610 frames, 27382 counted persons, native scale. Scout threshold 0.5; the 224 chains keep R19's confirmer; both 448 chains use R25b's, R19's recipe fitted on the 448 seed-0 scout's train candidates (seed 0 kept: the lowest val pass at 95% keep; seed 0 43.5%, seed 1 44.8%, seed 2 45.9%). Each confirmer's threshold is set per scout to keep 95% of that scout's val people.

| Seed | Camera | Candidates per frame | False candidates per frame | Confirmer AUC | People found | False alarms passed per frame | Share of false alarms passed |
|---|---|---|---|---|---|---|---|
| 0 | 224 px | 7.23 | 2.17 | 0.911 | 27.0% | 0.514 | 23.7% |
| 0 | 448 px | 7.06 | 1.70 | 0.859 | 28.7% | 0.647 | 38.0% |
| 1 | 224 px | 6.98 | 2.01 | 0.910 | 26.6% | 0.483 | 24.1% |
| 1 | 448 px | 7.34 | 1.89 | 0.871 | 29.3% | 0.673 | 35.6% |

- Seed 0: 448 over 224 false alarms passed 1.26x [1.15, 1.37], people found +1.7 points.
- Seed 1: 448 over 224 false alarms passed 1.39x [1.26, 1.53], people found +2.7 points.

Intervals: 95% frame bootstrap (2,000 resamples, the same frames for both cameras). Reproduce: `python scripts/r25_camera.py crops --split train`, `fit --seed 0|1|2`, `score --confirmer own`, `report-b`.
