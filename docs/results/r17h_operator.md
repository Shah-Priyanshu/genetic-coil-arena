# R17 step 6c: the recommended search system with the measured operator (R15e)

Read-out (fixed in advance): variants that drop the share found below 40% or push wrong finds above 5%: measured operator, clear cases (92% / 23%), measured operator, VisDrone labels (96% / 46%).

Step 3's 40 held-out missions × 2,000 episodes; every mission replanned at mission time for each variant; one assumption moved at a time from step 6's recommended system.

| Variant | Found | Change | Wrong finds |
|---|---|---|---|
| measured operator, VisDrone labels (96% / 46%) | 23.3% | -29.0 | 33.9% |
| measured operator, clear cases (92% / 23%) | 25.4% | -26.9 | 20.3% |
| recommended (as in step 6) | 52.3% | +0.0 | 4.8% |

Reproduce: `python scripts/r17h_sensitivity.py --from runs:r17b-train-cond-*`.
