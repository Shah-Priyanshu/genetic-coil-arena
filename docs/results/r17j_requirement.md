# R17 step 6h: the false-accept rate the evidence chain must reach

- **Cost 2:** the largest false-accept rate per candidate that meets the limits is 2.0% (0.41x R19's 4.9%): 51.9% found, 4.6% wrong finds.
- **Cost 3:** the largest false-accept rate per candidate that meets the limits is 3.0% (0.61x R19's 4.9%): 41.7% found, 4.8% wrong finds.

Limits: found >= 40%, wrong finds <= 5%. Step 3's 40 held-out missions x 2,000 test episodes; people kept 92% (R19, R20b); every mission tuned at mission time with found - cost x wrong; 95% CI over missions.

| Cost | False accept per candidate | Found | Wrong finds | Meets |
|---|---|---|---|---|
| 2 | 0.5% | 64.4% ± 8.4 | 2.6% ± 0.3 | yes |
| 2 | 1.0% | 58.2% ± 9.2 | 3.4% ± 0.4 | yes |
| 2 | 1.5% | 55.5% ± 9.4 | 4.0% ± 0.5 | yes |
| 2 | 2.0% | 51.9% ± 9.3 | 4.6% ± 0.6 | yes |
| 2 | 2.5% | 49.9% ± 9.6 | 5.2% ± 0.6 | no |
| 2 | 3.0% | 48.2% ± 9.7 | 5.6% ± 0.6 | no |
| 2 | 3.5% | 45.5% ± 9.3 | 6.3% ± 0.8 | no |
| 2 | 4.0% | 44.0% ± 9.2 | 6.6% ± 0.8 | no |
| 3 | 0.5% | 61.4% ± 9.1 | 2.4% ± 0.3 | yes |
| 3 | 1.0% | 55.2% ± 9.3 | 3.0% ± 0.4 | yes |
| 3 | 1.5% | 49.6% ± 10.2 | 3.5% ± 0.5 | yes |
| 3 | 2.0% | 46.7% ± 10.3 | 3.9% ± 0.5 | yes |
| 3 | 2.5% | 43.7% ± 10.0 | 4.4% ± 0.5 | yes |
| 3 | 3.0% | 41.7% ± 9.7 | 4.8% ± 0.5 | yes |
| 3 | 3.5% | 37.8% ± 9.5 | 5.1% ± 0.5 | no |
| 3 | 4.0% | 37.0% ± 9.5 | 5.7% ± 0.7 | no |

Reproduce: `python scripts/r17j_requirement.py --from runs:r17b-train-cond-*`.
