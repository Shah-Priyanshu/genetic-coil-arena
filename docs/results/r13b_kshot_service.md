# R13b probe: how many labelled incidents does the fault specialist need on a new system?

Split: `service`. Exploratory probe (not a pre-registered goal). RE1 (375 incidents, metrics only). The frozen G4c fault specialist plus a softmax adapter fitted on k labelled incidents of the target system (L2 towards the frozen model), tested on the rest. Mean ± 95% t-CI over 30 random draws of the k incidents. Joint = BARO's top service right and the fault type right.

| System | k | Adapted joint | Scratch joint | Frozen joint | BARO joint | Adapted fault acc. |
|---|---|---|---|---|---|---|
| OB (n = 125) | 0 | 54.3 ± 2.4 | 20.0 ± 0.0 | 54.3 ± 2.4 | 54.6 ± 1.6 | 66.5 ± 1.9 |
| OB (n = 125) | 5 | 54.5 ± 5.1 | 53.2 ± 5.1 | 52.4 ± 2.4 | 54.9 ± 1.7 | 63.6 ± 5.6 |
| OB (n = 125) | 10 | 69.0 ± 4.7 | 68.1 ± 4.9 | 52.6 ± 2.6 | 54.6 ± 1.7 | 81.1 ± 4.3 |
| OB (n = 125) | 20 | 78.5 ± 3.6 | 78.4 ± 3.8 | 51.9 ± 2.3 | 53.3 ± 1.7 | 92.0 ± 2.8 |
| OB (n = 125) | 40 | 86.1 ± 2.6 | 85.9 ± 2.7 | 53.9 ± 2.5 | 55.3 ± 1.6 | 97.7 ± 1.0 |
| SS (n = 125) | 0 | 58.3 ± 3.3 | 20.0 ± 0.0 | 58.3 ± 3.3 | 36.9 ± 3.6 | 65.4 ± 3.3 |
| SS (n = 125) | 5 | 46.6 ± 3.2 | 44.3 ± 3.5 | 55.3 ± 3.3 | 33.6 ± 3.3 | 52.4 ± 3.7 |
| SS (n = 125) | 10 | 56.7 ± 4.0 | 54.2 ± 4.1 | 54.9 ± 3.3 | 35.5 ± 3.8 | 62.8 ± 4.6 |
| SS (n = 125) | 20 | 58.9 ± 3.1 | 57.7 ± 2.8 | 56.9 ± 3.3 | 35.1 ± 4.7 | 68.5 ± 3.0 |
| SS (n = 125) | 40 | 61.9 ± 2.8 | 62.1 ± 3.1 | 54.8 ± 3.2 | 34.2 ± 4.0 | 71.3 ± 2.9 |
| TT (n = 125) | 0 | 31.3 ± 2.4 | 20.0 ± 0.0 | 31.3 ± 2.4 | 24.0 ± 1.0 | 45.5 ± 2.0 |
| TT (n = 125) | 5 | 36.2 ± 2.8 | 34.7 ± 3.2 | 32.0 ± 2.2 | 24.0 ± 1.1 | 51.5 ± 3.2 |
| TT (n = 125) | 10 | 44.2 ± 3.5 | 43.5 ± 3.6 | 32.0 ± 1.7 | 24.0 ± 0.9 | 61.7 ± 3.2 |
| TT (n = 125) | 20 | 49.3 ± 3.1 | 47.7 ± 3.1 | 31.5 ± 2.0 | 23.9 ± 1.2 | 69.1 ± 3.5 |
| TT (n = 125) | 40 | 53.5 ± 2.2 | 53.1 ± 2.0 | 30.7 ± 2.2 | 23.9 ± 0.9 | 75.9 ± 2.6 |

Adapter: input = 6 log-probabilities + 8 kind log z-scores + bias; L2 λ = 1.0; 400 gradient steps at 0.1. Frozen model SHA-256 `953524c1ce6905fc`.

Reproduce: `python scripts/r13b_kshot.py --team-from runs:r11-team`.
