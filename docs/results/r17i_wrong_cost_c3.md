# R17 step 6g, post-hoc: wrong-find cost 3

**Post-hoc point, not part of the pre-registered grid (2, 4, 8, 16):** added after the FAIL to see whether a cost between 2 and 4 meets both limits at the cut's upper bound. It does not (5.1% wrong finds).

Step 3's 40 held-out missions x 2,000 test episodes; each mission tuned at mission time with found - cost x wrong as the objective (the mission score is found - 2 x wrong); 95% CI over missions.

| Planned for | Flown in | Cost | Found | Wrong finds | Meets |
|---|---|---|---|---|---|
| measured cut (2.1%) | measured cut (2.1%) | 3 | 46.4% ± 10.0 | 4.2% ± 0.5 | yes |
| replicate's cut (1.3%) | replicate's cut (1.3%) | 3 | 52.6% ± 9.6 | 3.5% ± 0.4 | yes |
| cut's upper bound (3.2%) | measured cut (2.1%) | 3 | 42.7% ± 9.7 | 3.6% ± 0.4 | yes |
| cut's upper bound (3.2%) | replicate's cut (1.3%) | 3 | 43.1% ± 9.8 | 2.4% ± 0.3 | yes |
| cut's upper bound (3.2%) | cut's upper bound (3.2%) | 3 | 42.3% ± 9.5 | 5.1% ± 0.5 | no |
| R19 alone (4.9%) | R19 alone (4.9%) | 3 | 34.6% ± 8.9 | 6.6% ± 0.8 | no |

Reproduce: `python scripts/r17i_wrong_cost.py --from runs:r17b-train-cond-* --costs 3 --out docs/results/r17i_wrong_cost_c3.md`.
