# R17 step 6b: which assumption matters most?

Read-out (fixed in advance): variants that drop the share found below 40% or push wrong finds above 5%: R19: confirmer at the ground, then operator (92% / 5%), R20b: the same at the cut's upper bound (92% / 3.2%).

Step 3's 40 held-out missions × 2,000 episodes; every mission replanned at mission time for each variant; one assumption moved at a time from step 6's recommended system.

| Variant | Found | Change | Wrong finds |
|---|---|---|---|
| R19: confirmer at the ground, then operator (92% / 5%) | 40.4% | -11.9 | 7.0% |
| R20b: the same at the cut's upper bound (92% / 3.2%) | 47.6% | -4.7 | 5.6% |
| R20b: confirmer and closer look at the hover, then operator (92% / 2.1%) | 51.9% | -0.4 | 4.7% |
| recommended (as in step 6) | 52.3% | +0.0 | 4.8% |
| R20b: the same at the replicate's cut (92% / 1.3%) | 56.2% | +3.8 | 3.7% |

Reproduce: `python scripts/r17h_sensitivity.py --from runs:r17b-train-cond-*`.
