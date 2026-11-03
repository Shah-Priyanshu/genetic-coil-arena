# R17 step 6e: the search with the R19 confirmer (one rate changed at a time)

Read-out (fixed in advance): variants that drop the share found below 40% or push wrong finds above 5%: careful operator, clear cases (90% / 8%), R19: confirmer at the ground, then operator (92% / 5%), R19: confirmer before dispatch, no video filter (operator after it 99% / 80%), R19: video filter and confirmer before dispatch (operator after it 99% / 80%).

Step 3's 40 held-out missions × 2,000 episodes; every mission replanned at mission time for each variant; one assumption moved at a time from step 6's recommended system.

| Variant | Found | Change | Wrong finds |
|---|---|---|---|
| R19: confirmer before dispatch, no video filter (operator after it 99% / 80%) | 23.6% | -28.7 | 36.6% |
| careful operator, clear cases (90% / 8%) | 35.7% | -16.6 | 10.2% |
| R19: confirmer at the ground, then operator (92% / 5%) | 40.4% | -11.9 | 7.0% |
| R19: video filter and confirmer before dispatch (operator after it 99% / 80%) | 50.2% | -2.1 | 16.4% |
| recommended (as in step 6) | 52.3% | +0.0 | 4.8% |

Reproduce: `python scripts/r17h_sensitivity.py --from runs:r17b-train-cond-*`.
