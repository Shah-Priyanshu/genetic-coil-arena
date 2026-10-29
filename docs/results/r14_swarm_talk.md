# R14 step 1: does a drone swarm need to talk, and does its radio bind? (goal G13)

Generated 2026-10-29 by job `r14-swarm-talk` (pinned to 04aaf2ac).

**G13: RADIO DOES NOT BIND.** At flow-deck drift 0.03 m per step, the silent swarm confirms the source in 73.4% of episodes and the 8-byte lossless swarm in 97.6%: talking adds +24.2 ± 1.4 points (need ≥ 15). One byte per packet at 30% loss reaches 94.9% and keeps 89% of that gain (the radio binds if ≤ 67%).

Lab: `envs/swarm`, 3 gas and 3 camera drones, 30 m arena, source found when a camera is within 1.5 m, 240 steps of 0.5 s; one packet per drone per step (2 Hz). Mean ± 95% t-CI over replicates (same seeds in every cell).

<!-- analysis:start -->
**What step 1 shows: the swarm must talk, but two bytes say everything this task needs.**

- **Talk matters.** Camera drones cannot smell, so silent cameras only find the source by chance. Sharing the best gas reading adds +24.2 ± 1.4 points of success and finds the source about a third sooner: 94 steps instead of 140.
- **The radio does not bind.**
  - **2 bytes** (5 position bits per axis, about 1 m cells) already match 8 bytes.
  - **1 byte** (3 bits per axis, 3.75 m cells) keeps 89% of the success gain at 30% loss. It is about 13 steps slower, because the coarse cell must be searched.
  - So learned codes could win at most about 3 points of success, or some speed, at 1 byte. At ESP-NOW's 250 bytes per packet there is nothing to compress.
- **Loss and drift barely matter at this scale.** 30% packet loss costs nothing, because every drone repeats its best reading at 2 Hz. Tripling the flow drift (0.1 m per step, about 1.5 m after 2 minutes) costs 1–4 points.
- **The search radius mattered more than the code.** Every budget's tuning chose 4.5 m, the largest radius in the grid, so a wider grid might raise every talking row a little. That would not change the verdict.

**Reading for the direction.** A single "where is the strongest reading" report is a few bits of information. Hand codes carry it in 1–2 bytes, and realistic radios carry far more. Bandwidth binds only when what must be shared is rich: maps, images, many simultaneous targets, or very large swarms on one channel. So, for this swarm, the case for learning is not compression. It could be coordination that is hard to hand-design: role allocation among several targets, obstacles and failures. In this project's evidence, that is also where learned centres have only *matched* good hand designs (G7) so far.
<!-- analysis:end -->

## Success rate (%)

| Radio | loss 0, drift 0.03 | loss 0.3, drift 0.03 | loss 0, drift 0.1 | loss 0.3, drift 0.1 |
|---|---|---|---|---|
| silent | 73.4 ± 1.3 | 73.4 ± 1.3 | 70.4 ± 1.5 | 70.4 ± 1.5 |
| 1 B | 94.1 ± 1.0 | 94.9 ± 1.1 | 90.1 ± 1.2 | 90.5 ± 1.1 |
| 2 B | 97.5 ± 0.8 | 97.6 ± 0.6 | 96.4 ± 1.0 | 95.8 ± 0.7 |
| 4 B | 98.3 ± 0.6 | 98.3 ± 0.8 | 96.8 ± 0.9 | 96.5 ± 0.8 |
| 8 B | 97.6 ± 0.5 | 97.8 ± 0.5 | 96.2 ± 0.7 | 96.4 ± 1.1 |

## Steps to confirm (censored at 240)

| Radio | loss 0, drift 0.03 | loss 0.3, drift 0.03 | loss 0, drift 0.1 | loss 0.3, drift 0.1 |
|---|---|---|---|---|
| silent | 139.5 ± 2.9 | 139.5 ± 2.9 | 143.8 ± 2.8 | 143.8 ± 2.8 |
| 1 B | 108.7 ± 2.5 | 106.7 ± 3.7 | 113.6 ± 3.1 | 113.2 ± 3.3 |
| 2 B | 94.3 ± 2.9 | 94.9 ± 2.4 | 99.6 ± 2.6 | 100.7 ± 2.5 |
| 4 B | 93.2 ± 2.3 | 93.1 ± 3.6 | 98.6 ± 3.0 | 99.1 ± 3.3 |
| 8 B | 94.5 ± 2.5 | 94.5 ± 2.6 | 98.6 ± 2.1 | 99.1 ± 3.2 |

## Share of the talk gain kept (drift 0.03)

| Budget | loss 0 | loss 0.3 |
|---|---|---|
| 1 B | 86% | 89% |
| 2 B | 100% | 100% |
| 4 B | 103% | 103% |
| 8 B | 100% | 101% |

Hand codes tuned on a separate seed block at loss 0.3, drift 0.03: 1 B: 3 position bits per axis, search radius 4.5 m (tuning success 93.2%); 2 B: 5 position bits per axis, search radius 4.5 m (tuning success 98.3%); 4 B: 8 position bits per axis, search radius 4.5 m (tuning success 98.7%); 8 B: 24 position bits per axis, search radius 4.5 m (tuning success 98.5%).

Reproduce: `python scripts/r14_swarm_talk.py`.
