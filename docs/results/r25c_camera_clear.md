# R25c: the 448-px camera, counting only clear non-persons

**NOT PASSED.** Pooled over seeds, the 448 chain passes 0.77x [0.46, 1.40] the 224 chain's clear non-persons per test-dev frame (R25's bars: pooled <= 0.67x with the upper bound below 1, each seed <= 0.8x, people found within 1 point; KILL at >= 0.9x).

R25b's chains. Per chain 400 of its passed test-dev false alarms (VisDrone labels) drawn at random (seed 2025) and labelled blind by Claude subagents, chain hidden; clear non-persons per frame = passed false alarms per frame x the share labelled not a person.

| Chain | Confirmer | Passed false alarms per frame | Sampled | Person | Unclear | Not a person | Clear non-persons per frame |
|---|---|---|---|---|---|---|---|
| 224 px, seed0 | R19's | 0.514 | 400 | 337 | 42 | 21 | 0.027 |
| 448 px, seed0 | its own (R25b) | 0.647 | 400 | 359 | 29 | 12 | 0.019 |
| 224 px, seed1 | R19's | 0.483 | 400 | 329 | 51 | 20 | 0.024 |
| 448 px, seed1 | its own (R25b) | 0.673 | 400 | 349 | 39 | 12 | 0.020 |

- Seed 0: 448 over 224 clear non-persons passed 0.72x [0.31, 1.55]; people found +1.7 points (R25b).
- Seed 1: 448 over 224 clear non-persons passed 0.84x [0.39, 1.69]; people found +2.7 points (R25b).

Repeated spots labelled the same both times: 31 of 36. Intervals: 95% frame bootstrap (2,000 resamples, the same frames for every chain), which also resamples the labelled spots. Reproduce: `python scripts/r25_camera.py sample-c`, label the sheets, `report-c`.
