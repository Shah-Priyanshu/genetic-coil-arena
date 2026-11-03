# R17 step 6d: the recommended search system with the careful operator (R15f)

Read-out (fixed in advance): variants that drop the share found below 40% or push wrong finds above 5%: measured operator, clear cases (92% / 23%), careful operator, clear cases (90% / 8%), careful operator, VisDrone labels (99% / 36%).

Step 3's 40 held-out missions × 2,000 episodes; every mission replanned at mission time for each variant; one assumption moved at a time from step 6's recommended system.

| Variant | Found | Change | Wrong finds |
|---|---|---|---|
| careful operator, VisDrone labels (99% / 36%) | 24.8% | -27.5 | 29.1% |
| measured operator, clear cases (92% / 23%) | 25.4% | -26.9 | 20.3% |
| careful operator, clear cases (90% / 8%) | 35.7% | -16.6 | 10.2% |
| recommended (as in step 6) | 52.3% | +0.0 | 4.8% |

Reproduce: `python scripts/r17h_sensitivity.py --from runs:r17b-train-cond-* --variants "recommended (as in step 6)" "measured operator, clear cases (92% / 23%)" "careful operator, clear cases (90% / 8%)" "careful operator, VisDrone labels (99% / 36%)" --out docs/results/r17h_careful.md`. The quick-look row reproduces R17 step 6c exactly.
