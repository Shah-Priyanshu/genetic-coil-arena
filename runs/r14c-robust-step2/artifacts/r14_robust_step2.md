# R14 step 3: do the learned swarm brains survive worse conditions?

**3a, robust: PASS.** The step-2 brains beat the hand design with the replicate CI above 0 in every stress condition.

All three leaks found (3 sources, 3000 episodes per cell; gains are replicate means ± 95% CI).

| Condition | Hand design | step2 (gain over hand) | Sensed oracle |
|---|---|---|---|
| nominal | 62.5% | 78.3% (+15.9 ± 2.0) | 93.2% |
| loss 50% | 63.4% | 78.0% (+14.6 ± 1.7) | 93.2% |
| loss 70% | 60.4% | 77.3% (+17.0 ± 2.4) | 93.2% |
| drift x3 | 51.4% | 73.0% (+21.6 ± 2.5) | 91.8% |
| weak gas sensor | 50.9% | 75.1% (+24.2 ± 1.8) | 92.0% |
| narrow plume | 66.2% | 71.6% (+5.4 ± 1.7) | 82.0% |
| wide plume | 10.4% | 67.5% (+57.1 ± 1.5) | 46.5% |
| 1 drone down | 54.0% | 71.4% (+17.4 ± 2.4) | 90.5% |
| 2 drones down | 40.7% | 61.6% (+21.0 ± 2.5) | 86.0% |
| 4 drones (2+2) * | 36.9% | 56.5% (+19.6 ± 1.8) | 82.6% |
| 40 m arena * | 30.3% | 45.6% (+15.3 ± 1.3) | 64.2% |

\* outside the randomised training range (and the step-2 setting).

Sets: `step2` (16 training runs). `step2` = trained at the nominal setting only (R14 step 2); `random` = a random condition every generation, same budget.

Reproduce: `python scripts/r14_robust.py --set step2=runs:r14b-train-* --set random=runs:r14c-train-*`.
