# Squad: need to talk

- Date: 2026-10-23
- Commit: 5d64e76
- Seeds: 100 held-out episodes per row, seeds 2000000000..2000000099
- CPU: Intel(R) Xeon(R) Processor @ 2.80GHz (4 logical cores visible; throughput is one process on one core)
- Python 3.13.16
- Squad defaults: 21×21 walls, 3 snakes, 2 food per colour, health 60, T = 300, message = 1 discrete symbol, V = 8, every step

Team food per episode (all snakes), mean ± 95% CI over episodes.

| r | Setting | Team food | Survivors at end | Episode steps |
|---|---|---|---|---|
| 3 | greedy + hand-coded centre (3 bits) | 17.29 ± 1.79 | 0.56 ± 0.11 | 246.79 ± 15.47 |
| 3 | greedy + silent centre | 7.10 ± 0.85 | 0.09 ± 0.06 | 162.82 ± 14.09 |
| 3 | greedy + random centre | 3.75 ± 0.48 | 0.00 ± 0.00 | 110.52 ± 7.99 |
| 3 | no-comm greedy (ignores messages) | 8.00 ± 1.14 | 0.12 ± 0.06 | 170.31 ± 15.78 |
| 5 | greedy + hand-coded centre (3 bits) | 27.67 ± 1.78 | 0.74 ± 0.13 | 273.11 ± 10.00 |
| 5 | greedy + silent centre | 15.74 ± 1.55 | 0.39 ± 0.10 | 228.55 ± 14.90 |
| 5 | greedy + random centre | 7.56 ± 0.86 | 0.01 ± 0.02 | 136.38 ± 9.98 |
| 5 | no-comm greedy (ignores messages) | 15.97 ± 1.61 | 0.31 ± 0.10 | 222.57 ± 14.95 |
| 7 | greedy + hand-coded centre (3 bits) | 31.95 ± 1.85 | 0.80 ± 0.13 | 277.37 ± 8.86 |
| 7 | greedy + silent centre | 24.67 ± 1.84 | 0.72 ± 0.13 | 261.95 ± 12.92 |
| 7 | greedy + random centre | 14.69 ± 1.46 | 0.06 ± 0.05 | 169.60 ± 12.77 |
| 7 | no-comm greedy (ignores messages) | 24.29 ± 1.88 | 0.48 ± 0.12 | 247.97 ± 12.67 |
| full | oracle (specialists see the full board) | 41.69 ± 2.22 | 0.77 ± 0.12 | 275.20 ± 9.33 |
| - | random actions | 0.39 ± 0.13 | 0.00 ± 0.00 | 52.21 ± 3.74 |

Throughput (batch 256, best of 3, living-snake steps per second, one process):

| Rollout | Agent-steps/s |
|---|---|
| greedy + hand-coded centre | 579,900 |
| oracle | 693,347 |
| engine only (random actions) | 604,045 |

Notes:

- Reproduce with `python scripts/need_to_talk.py --seeds 100 --out docs/results/need_to_talk.md`.
- "Greedy" specialists chase the nearest own-colour food in their window. With no food in
  view they follow the message's direction bits. A silent centre leaves the payload at
  code 0 ("ahead, near"), so "greedy + silent" is a mostly-straight wander. "No-comm"
  wanders with random turns instead. Both land in the same place.
- Messages are one step stale: they are encoded against the receiver's heading at send time
  and read after its next move.
- The 3-bit hand-coded channel recovers most of the gap between the window-only baselines
  and the full-view oracle at r = 3 and r = 5. The gap narrows as r grows, because larger
  windows already show more food.
