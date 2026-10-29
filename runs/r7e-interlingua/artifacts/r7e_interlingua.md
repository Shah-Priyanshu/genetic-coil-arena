# R7e probe: independently trained squads through a shared meaning space

Exploratory probe (not a pre-registered goal). Squad A's centre drives squad B's specialists on the 6 held-out goals; the bridge reads A's symbols into eat/do-not-eat with A's G12 decoder and writes them in B's dialect with B's G12 writer. 34 pairs (each G12-sample run with the next), 50 episodes per goal. Mean ± 95% t-CI over pairs.

| Condition | Mission score | Share of B's native score |
|---|---|---|
| B's own centre (native) | 21.63 ± 0.56 | 100% ± 0 |
| A's centre, no bridge | -0.07 ± 2.03 | -0% ± 9 |
| A's centre through the bridge | 15.35 ± 1.79 | 71% ± 8 |
| hand-coded centre written in B's dialect | 15.81 ± 1.79 | 73% ± 8 |
| true status written in B's dialect | 15.81 ± 1.79 | 73% ± 8 |

Bridge gain over no bridge: +15.42 ± 2.67.

Reproduce: `python scripts/r7e_interlingua.py --from-json runs:r7b3-train-* runs:r7b4-train-* runs:r7b5-train-* runs:r7c-train-*`.
