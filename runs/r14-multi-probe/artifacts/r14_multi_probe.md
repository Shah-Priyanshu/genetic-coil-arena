# R14 probe: several gas sources, how much room does coordination leave?

Exploratory probe (not a pre-registered goal). 3 gas and 3 camera drones, 30 m arena, 240 steps (2 min), 30% packet loss, drift 0.03 m per step. Mean ± 95% t-CI over 10 replicates of 300 episodes (same seeds for every system).

| Sources | System | Sources found | All found | Steps |
|---|---|---|---|---|
| 1 | silent | 0.71 ± 0.01 | 71.5 ± 1.0% | 144 ± 2 |
| 1 | hand 2B | 0.97 ± 0.01 | 96.6 ± 0.7% | 102 ± 2 |
| 1 | hand 8B | 0.97 ± 0.01 | 96.9 ± 0.6% | 103 ± 1 |
| 1 | sensed oracle | 0.99 ± 0.01 | 98.6 ± 0.6% | 70 ± 2 |
| 1 | oracle | 1.00 ± 0.00 | 100.0 ± 0.0% | 46 ± 0 |
| 3 | silent | 2.05 ± 0.03 | 29.9 ± 2.1% | 218 ± 2 |
| 3 | hand 2B | 2.58 ± 0.02 | 61.9 ± 2.5% | 201 ± 2 |
| 3 | hand 8B | 2.59 ± 0.03 | 62.2 ± 2.6% | 199 ± 2 |
| 3 | sensed oracle | 2.93 ± 0.01 | 93.1 ± 1.0% | 124 ± 2 |
| 3 | oracle | 3.00 ± 0.00 | 100.0 ± 0.0% | 59 ± 0 |

Differences, paired by replicate:

- K = 1, sensed oracle minus hand 2 B: found +0.02 ± 0.00; all +2.03 ± 0.46 points; steps -31.81 ± 1.63 steps.
- K = 1, silent minus hand 2 B: found -0.25 ± 0.01; all -25.10 ± 1.21 points; steps +41.77 ± 2.88 steps.
- K = 3, sensed oracle minus hand 2 B: found +0.35 ± 0.03; all +31.27 ± 2.60 points; steps -77.09 ± 2.08 steps.
- K = 3, silent minus hand 2 B: found -0.53 ± 0.02; all -32.00 ± 2.17 points; steps +16.28 ± 1.93 steps.

Hand knobs tuned at K = 3 on a separate block: one camera per hotspot False, camera radius 4.5 m, exclusion 8 m (tuning: 2.57 sources found).

Reproduce: `python scripts/r14_multi_probe.py`.
