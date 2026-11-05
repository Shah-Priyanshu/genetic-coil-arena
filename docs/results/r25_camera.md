# R25: does the 448-px camera cut the false alarms that pass the confirmer?

**KILL.** Pooled over seeds, the 448 chain passes 1.42x [1.28, 1.56] the 224 chain's labelled false alarms per test-dev frame (bars: pooled <= 0.67x with the upper bound below 1, each seed <= 0.8x, people found within 1 point; KILL at >= 0.9x).

VisDrone2019-DET test-dev, 1610 frames, 27382 counted persons, native scale. Scout threshold 0.5; R19's confirmer (DINOv2-small fine-tuned) for every chain. Each confirmer's threshold is set per scout to keep 95% of that scout's val people.

| Seed | Camera | Candidates per frame | False candidates per frame | Confirmer AUC | People found | False alarms passed per frame | Share of false alarms passed |
|---|---|---|---|---|---|---|---|
| 0 | 224 px | 7.23 | 2.17 | 0.911 | 27.0% | 0.514 | 23.7% |
| 0 | 448 px | 7.06 | 1.70 | 0.847 | 29.1% | 0.688 | 40.5% |
| 1 | 224 px | 6.98 | 2.01 | 0.910 | 26.6% | 0.483 | 24.1% |
| 1 | 448 px | 7.34 | 1.89 | 0.858 | 29.6% | 0.725 | 38.4% |

- Seed 0: 448 over 224 false alarms passed 1.34x [1.20, 1.48], people found +2.1 points.
- Seed 1: 448 over 224 false alarms passed 1.50x [1.35, 1.66], people found +3.0 points.

Intervals: 95% frame bootstrap (2,000 resamples, the same frames for both cameras). Reproduce: `python scripts/r25_camera.py crops`, `score`, `report`.
