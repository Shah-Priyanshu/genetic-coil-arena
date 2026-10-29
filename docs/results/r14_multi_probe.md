# R14 probe: several gas sources, how much room does coordination leave?

Exploratory probe (not a pre-registered goal). 3 gas and 3 camera drones, 30 m arena, 240 steps (2 min), 30% packet loss, drift 0.03 m per step. Mean ± 95% t-CI over 10 replicates of 300 episodes (same seeds for every system).

<!-- analysis:start -->
**What the probe shows: with several targets, coordination becomes the bottleneck, and a hand design leaves a lot on the table.**

- **One source: little room.** The tuned hand swarm finds it 96.6% of the time. Perfect sharing of what the swarm smells (the sensed oracle) adds only 2 points, though it is 32 steps faster.
- **Three sources: large room.**

  | System | All three found |
  |---|---|
  | Tuned hand design | 61.9% |
  | Sensed oracle | 93.1% |
  | Silent | 29.9% |

  The gap of +31.3 ± 2.6 points is paired over the same episodes, and the sensed oracle is also 77 steps faster. That gap is what better sharing, allocation and re-tasking could add *without any better sensing*.
- **The radio still does not bind.** 8 bytes equal 2 bytes in every row, so the room is in *what the drones do with* the information: who goes where, when to move on. Squeezing it into fewer bits does not help.
- **Hand design was fragile here.** A first reasonable gossip design was worse than silence (1.89 against 2.03 sources in the pilot). Tuning one knob, how far to stay away from a confirmed source, lifted it to 2.58.
- **Caveats.**
  - The sensed oracle learns a source's *exact* position from the first sniff, so part of the gap is localisation, not coordination.
  - The hand design was tuned over 12 settings only. A better hand design could close some of the gap.
  - Whether a *learned* policy closes it is untested; that needs ES training (about a day of CPU).
<!-- analysis:end -->

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
