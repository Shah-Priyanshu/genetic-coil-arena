# R20: a closer look from the confirming drone against a second model on the scout's view

**KILL.** On test-dev false alarms, the closer-look chain passes 0.86x [0.82, 0.90] of what the control chain passes (bar: <= 0.5x with the bootstrap upper bound below 1, replicated, and >= 90% of test-dev people kept; kill at >= 0.8x).

Scout's view: VisDrone2019-DET reduced 2x (the scout twice as high); closer look: native resolution. 6916 val and 9366 test-dev candidates (6183 people, 3183 false alarms by VisDrone's labels). DINOv2-small fine-tuned per view (R19's recipe), thresholds keep 95% of val people.

| Model | Val AUC | Val false alarms passed | Test-dev people kept | Test-dev false alarms passed |
|---|---|---|---|---|
| far_s0 | 0.916 | 28.0% | 93.4% [93, 94] | 23.1% [22, 25] |
| far_s1 | 0.919 | 28.4% | 93.5% [93, 94] | 21.9% [20, 23] |
| close_s0 | 0.940 | 20.7% | 93.7% [93, 94] | 18.0% [17, 19] |
| close_s1 | 0.938 | 21.0% | 93.6% [93, 94] | 17.8% [16, 19] |

| Chain (both must pass) | Val people kept | Val false alarms passed | Test-dev people kept | Test-dev false alarms passed |
|---|---|---|---|---|
| scout's view, checked by two models | 95.0% | 27.7% | 93.6% [93, 94] | 21.2% [20, 23] |
| scout's view, then the closer look | 95.0% | 22.5% | 93.6% [93, 94] | 18.3% [17, 20] |
| the same, other seeds | 95.0% | 22.0% | 93.4% [93, 94] | 17.8% [16, 19] |

Second model's errors on test-dev false alarms, against far seed 0's (each model at its own 95% threshold):

| Second model | Passes alone | Passes when far s0 passed | Lift over independent | Score rank correlation |
|---|---|---|---|---|
| far_s1 | 21.9% | 82.7% | 3.78x | 0.94 |
| close_s0 | 18.0% | 66.9% | 3.72x | 0.85 |

Blind visual check of 150 test-dev false alarms passed by either chain: 12 clear non-persons, 111 unlabelled people, 27 ambiguous. Clear non-persons passed: control 12, closer look 3 (0.25x).

## R20b: every other passed false alarm, checked blind

**PASS.** On the 41 clear non-persons among 594 test-dev false alarms R20 did not check, the closer-look chain passes 0.43x [0.24, 0.66] of what the control passes (replicate 0.26x [0.10, 0.43]; bar: <= 0.5x with the upper bound below 1 and the replicate <= 0.5x; kill at >= 0.8x).

The 594 spots: 457 people, 96 unclear, 41 clearly not a person.

| Chain | Clear non-persons passed | People passed | Unclear passed | People among its passed spots |
|---|---|---|---|---|
| scout's view, checked by two models | 35 | 422 | 77 | 79% |
| scout's view, then the closer look | 15 | 395 | 52 | 85% |
| the same, other seeds | 9 | 389 | 58 | 85% |

Agreement with R20's check on 36 spots mixed in: 81% of labels; R20's clear non-persons kept 0 of 1, its people 26 of 28.

Reproduce: `python scripts/r20_closer_look.py report`.
