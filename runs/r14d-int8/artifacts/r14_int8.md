# R14 step 4a: the swarm brains in int8

**PASS.** At nominal the int8 brains find all leaks within 1 point of the float brains and still beat the hand design (replicate CI above 0).

All three leaks found, 3 sources, 3000 episodes per cell (int8 − float: replicate mean ± 95% CI).

| Condition | Hand | step2 float | step2 int8 (int8 − float) | random float | random int8 (int8 − float) |
|---|---|---|---|---|---|
| nominal | 62.5% | 78.3% | 78.2% (-0.1 ± 0.6) | 78.3% | 78.7% (+0.3 ± 0.6) |
| loss 70% | 60.4% | 77.3% | 77.4% (+0.1 ± 0.2) | 77.8% | 77.6% (-0.2 ± 0.6) |
| 2 drones down | 40.7% | 61.6% | 61.8% (+0.2 ± 0.6) | 61.8% | 61.6% (-0.3 ± 0.9) |

Reproduce: `python scripts/r14_int8.py --set step2=runs:r14b-train-* --set random=runs:r14c-train-*`.
