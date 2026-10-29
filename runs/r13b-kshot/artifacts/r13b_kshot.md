# R13b probe: how many labelled incidents does the fault specialist need on a new system?

Exploratory probe (not a pre-registered goal). RE1 (375 incidents, metrics only). The frozen G4c fault specialist plus a softmax adapter fitted on k labelled incidents of the target system (L2 towards the frozen model), tested on the rest. Mean ± 95% t-CI over 30 random draws of the k incidents. Joint = BARO's top service right and the fault type right.

| System | k | Adapted joint | Scratch joint | Frozen joint | BARO joint | Adapted fault acc. |
|---|---|---|---|---|---|---|
| OB (n = 125) | 0 | 52.8 ± 0.0 | 20.0 ± 0.0 | 52.8 ± 0.0 | 54.4 ± 0.0 | 65.6 ± 0.0 |
| OB (n = 125) | 5 | 53.6 ± 3.7 | 52.7 ± 3.7 | 52.8 ± 0.3 | 54.4 ± 0.3 | 62.2 ± 4.4 |
| OB (n = 125) | 10 | 69.4 ± 3.8 | 68.2 ± 3.7 | 53.1 ± 0.5 | 54.3 ± 0.5 | 79.8 ± 4.4 |
| OB (n = 125) | 20 | 83.3 ± 1.4 | 82.9 ± 1.6 | 52.9 ± 0.5 | 54.3 ± 0.6 | 96.2 ± 1.4 |
| OB (n = 125) | 40 | 85.6 ± 1.0 | 85.6 ± 1.0 | 52.3 ± 1.3 | 54.5 ± 1.1 | 98.7 ± 0.7 |
| SS (n = 125) | 0 | 55.2 ± 0.0 | 20.0 ± 0.0 | 55.2 ± 0.0 | 33.6 ± 0.0 | 63.2 ± 0.0 |
| SS (n = 125) | 5 | 47.1 ± 3.2 | 45.7 ± 3.4 | 55.3 ± 0.3 | 33.6 ± 0.2 | 53.5 ± 3.8 |
| SS (n = 125) | 10 | 56.8 ± 2.6 | 55.3 ± 3.0 | 55.3 ± 0.4 | 33.6 ± 0.5 | 65.0 ± 3.0 |
| SS (n = 125) | 20 | 63.0 ± 2.1 | 62.8 ± 2.3 | 55.4 ± 0.6 | 33.9 ± 0.7 | 71.8 ± 2.3 |
| SS (n = 125) | 40 | 66.4 ± 1.6 | 67.2 ± 1.9 | 54.7 ± 1.2 | 33.7 ± 1.2 | 75.6 ± 1.6 |
| TT (n = 125) | 0 | 32.0 ± 0.0 | 20.0 ± 0.0 | 32.0 ± 0.0 | 24.0 ± 0.0 | 45.6 ± 0.0 |
| TT (n = 125) | 5 | 34.6 ± 2.9 | 33.3 ± 2.9 | 32.1 ± 0.4 | 24.0 ± 0.4 | 49.1 ± 3.5 |
| TT (n = 125) | 10 | 47.2 ± 2.3 | 46.0 ± 2.4 | 31.9 ± 0.5 | 23.8 ± 0.5 | 66.6 ± 3.0 |
| TT (n = 125) | 20 | 49.5 ± 2.1 | 48.1 ± 2.0 | 31.7 ± 0.7 | 23.6 ± 0.6 | 69.4 ± 2.6 |
| TT (n = 125) | 40 | 54.0 ± 1.7 | 53.3 ± 2.0 | 32.0 ± 1.0 | 23.7 ± 0.9 | 76.2 ± 1.9 |

Adapter: input = 6 log-probabilities + 8 kind log z-scores + bias; L2 λ = 1.0; 400 gradient steps at 0.1. Frozen model SHA-256 `953524c1ce6905fc`.

Reproduce: `python scripts/r13b_kshot.py --team-from runs:r11-team`.
