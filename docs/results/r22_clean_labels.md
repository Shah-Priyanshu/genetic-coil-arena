# R22: the scout brain trained on confirmer-cleaned labels

**NOT PASSED.** Clear non-person false alarms per test-dev frame, cleaned scouts over control scouts (both seeds pooled): 0.89x [0.71, 1.11]; seed 0 0.96x, seed 1 0.83x (bar: <= 0.7x with the upper bound below 1, each seed <= 0.8x, people found within 2 points; kill at >= 0.9x).

Pseudo-labels: 2456 of 25495 unlabelled off-box train candidates became person boxes, in 1755 frames.

- blind check, pseudo positives: 77 people, 16 unclear, 7 clearly not a person
- blind check, rejected: 9 people, 44 unclear, 47 clearly not a person

| Scout | Threshold | Val people found | Test-dev people found | Test-dev false alarms per frame | Clear non-persons per frame |
|---|---|---|---|---|---|
| control_s0 | 0.500 | 44.1% | 29.7% [29, 30] | 2.17 | 0.74 |
| control_s1 | 0.495 | 44.4% | 29.9% [29, 30] | 2.17 | 0.72 |
| clean_s0 | 0.505 | 44.1% | 29.6% [29, 30] | 2.12 | 0.71 |
| clean_s1 | 0.500 | 44.5% | 29.5% [29, 30] | 2.14 | 0.60 |

Blind check of each scout's test-dev false alarms (VisDrone labels):

| Scout | People | Unclear | Clearly not a person |
|---|---|---|---|
| control_s0 | 56 | 43 | 51 |
| control_s1 | 57 | 43 | 50 |
| clean_s0 | 62 | 38 | 50 |
| clean_s1 | 79 | 29 | 42 |

Reproduce: `python scripts/r22_clean_labels.py report`.
