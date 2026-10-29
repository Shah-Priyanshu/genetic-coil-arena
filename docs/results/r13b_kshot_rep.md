# R13b probe: how many labelled incidents does the fault specialist need on a new system?

Split: `rep`. Exploratory probe (not a pre-registered goal). RE1 (375 incidents, metrics only). The frozen G4c fault specialist plus a softmax adapter fitted on k labelled incidents of the target system (L2 towards the frozen model), tested on the rest. Mean ± 95% t-CI over 30 random draws of the k incidents. Joint = BARO's top service right and the fault type right.

| System | k | Adapted joint | Scratch joint | Frozen joint | BARO joint | Adapted fault acc. |
|---|---|---|---|---|---|---|
| OB (n = 125) | 0 | 52.0 ± 0.0 | 20.0 ± 0.0 | 52.0 ± 0.0 | 55.0 ± 0.0 | 65.0 ± 0.0 |
| OB (n = 125) | 5 | 57.9 ± 4.2 | 57.6 ± 4.2 | 52.0 ± 0.0 | 55.0 ± 0.0 | 66.3 ± 4.6 |
| OB (n = 125) | 10 | 78.2 ± 3.4 | 76.4 ± 3.4 | 52.0 ± 0.0 | 55.0 ± 0.0 | 90.1 ± 3.8 |
| OB (n = 125) | 20 | 86.4 ± 0.5 | 86.1 ± 0.6 | 52.0 ± 0.0 | 55.0 ± 0.0 | 99.4 ± 0.5 |
| OB (n = 125) | 40 | 87.0 ± 0.0 | 87.0 ± 0.0 | 52.0 ± 0.0 | 55.0 ± 0.0 | 100.0 ± 0.0 |
| SS (n = 125) | 0 | 56.0 ± 0.0 | 20.0 ± 0.0 | 56.0 ± 0.0 | 34.0 ± 0.0 | 62.0 ± 0.0 |
| SS (n = 125) | 5 | 49.7 ± 3.9 | 49.2 ± 3.9 | 56.0 ± 0.0 | 34.0 ± 0.0 | 54.1 ± 4.4 |
| SS (n = 125) | 10 | 63.3 ± 2.3 | 63.2 ± 2.4 | 56.0 ± 0.0 | 34.0 ± 0.0 | 69.2 ± 2.6 |
| SS (n = 125) | 20 | 68.3 ± 1.0 | 69.3 ± 1.3 | 56.0 ± 0.0 | 34.0 ± 0.0 | 74.5 ± 1.0 |
| SS (n = 125) | 40 | 72.0 ± 0.0 | 73.0 ± 0.0 | 56.0 ± 0.0 | 34.0 ± 0.0 | 78.0 ± 0.0 |
| TT (n = 125) | 0 | 32.0 ± 0.0 | 20.0 ± 0.0 | 32.0 ± 0.0 | 24.0 ± 0.0 | 46.0 ± 0.0 |
| TT (n = 125) | 5 | 40.0 ± 3.3 | 38.6 ± 3.3 | 32.0 ± 0.0 | 24.0 ± 0.0 | 54.8 ± 4.6 |
| TT (n = 125) | 10 | 49.1 ± 2.2 | 46.6 ± 2.3 | 32.0 ± 0.0 | 24.0 ± 0.0 | 69.3 ± 2.5 |
| TT (n = 125) | 20 | 57.1 ± 1.3 | 55.2 ± 1.4 | 32.0 ± 0.0 | 24.0 ± 0.0 | 81.1 ± 1.7 |
| TT (n = 125) | 40 | 59.0 ± 0.0 | 59.0 ± 0.0 | 32.0 ± 0.0 | 24.0 ± 0.0 | 84.0 ± 0.0 |

Adapter: input = 6 log-probabilities + 8 kind log z-scores + bias; L2 λ = 1.0; 400 gradient steps at 0.1. Frozen model SHA-256 `953524c1ce6905fc`.

Reproduce: `python scripts/r13b_kshot.py --team-from runs:r11-team`.
