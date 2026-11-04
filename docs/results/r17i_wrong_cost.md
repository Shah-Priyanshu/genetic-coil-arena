# R17 step 6g: plan to the wrong-find limit

**Verdict: FAIL.** W* (smallest wrong-find cost at which the upper bound's wrong finds are <= 5%): 4.

- upper bound planned and flown: misses the limits (found >= 40%, wrong <= 5%)
- measured cut planned and flown: meets the limits (found >= 40%, wrong <= 5%)
- upper bound's plans flown at the measured cut: misses the limits (found >= 40%, wrong <= 5%)

Step 3's 40 held-out missions x 2,000 test episodes; each mission tuned at mission time with found - cost x wrong as the objective (the mission score is found - 2 x wrong); 95% CI over missions.

| Planned for | Flown in | Cost | Found | Wrong finds | Meets |
|---|---|---|---|---|---|
| measured cut (2.1%) | measured cut (2.1%) | 2 | 51.9% ± 9.7 | 4.7% ± 0.6 | yes |
| measured cut (2.1%) | measured cut (2.1%) | 4 | 41.1% ± 10.1 | 3.6% ± 0.4 | yes |
| measured cut (2.1%) | measured cut (2.1%) | 8 | 29.2% ± 8.1 | 2.7% ± 0.3 | no |
| measured cut (2.1%) | measured cut (2.1%) | 16 | 23.3% ± 7.6 | 2.5% ± 0.3 | no |
| replicate's cut (1.3%) | replicate's cut (1.3%) | 2 | 56.2% ± 9.2 | 3.7% ± 0.4 | yes |
| replicate's cut (1.3%) | replicate's cut (1.3%) | 4 | 49.0% ± 10.1 | 3.1% ± 0.4 | yes |
| replicate's cut (1.3%) | replicate's cut (1.3%) | 8 | 35.2% ± 9.6 | 2.2% ± 0.3 | no |
| replicate's cut (1.3%) | replicate's cut (1.3%) | 16 | 26.1% ± 7.8 | 1.8% ± 0.2 | no |
| cut's upper bound (3.2%) | measured cut (2.1%) | 2 | 48.2% ± 9.7 | 3.9% ± 0.4 | yes |
| cut's upper bound (3.2%) | measured cut (2.1%) | 4 | 36.6% ± 9.6 | 3.1% ± 0.3 | no |
| cut's upper bound (3.2%) | measured cut (2.1%) | 8 | 26.1% ± 7.9 | 2.6% ± 0.3 | no |
| cut's upper bound (3.2%) | measured cut (2.1%) | 16 | 21.3% ± 7.1 | 2.3% ± 0.3 | no |
| cut's upper bound (3.2%) | replicate's cut (1.3%) | 2 | 48.6% ± 9.8 | 2.7% ± 0.3 | yes |
| cut's upper bound (3.2%) | replicate's cut (1.3%) | 4 | 36.8% ± 9.7 | 2.1% ± 0.2 | no |
| cut's upper bound (3.2%) | replicate's cut (1.3%) | 8 | 26.1% ± 8.0 | 1.7% ± 0.2 | no |
| cut's upper bound (3.2%) | replicate's cut (1.3%) | 16 | 21.4% ± 7.2 | 1.6% ± 0.2 | no |
| cut's upper bound (3.2%) | cut's upper bound (3.2%) | 2 | 47.6% ± 9.5 | 5.6% ± 0.6 | no |
| cut's upper bound (3.2%) | cut's upper bound (3.2%) | 4 | 36.4% ± 9.5 | 4.5% ± 0.5 | no |
| cut's upper bound (3.2%) | cut's upper bound (3.2%) | 8 | 26.0% ± 7.9 | 3.8% ± 0.4 | no |
| cut's upper bound (3.2%) | cut's upper bound (3.2%) | 16 | 21.3% ± 7.1 | 3.4% ± 0.5 | no |
| R19 alone (4.9%) | R19 alone (4.9%) | 2 | 40.4% ± 9.2 | 7.0% ± 0.8 | no |
| R19 alone (4.9%) | R19 alone (4.9%) | 4 | 28.5% ± 8.2 | 5.6% ± 0.6 | no |
| R19 alone (4.9%) | R19 alone (4.9%) | 8 | 24.2% ± 7.7 | 5.7% ± 0.8 | no |
| R19 alone (4.9%) | R19 alone (4.9%) | 16 | 22.8% ± 7.8 | 5.5% ± 0.8 | no |

Reproduce: `python scripts/r17i_wrong_cost.py --from runs:r17b-train-cond-*`.
