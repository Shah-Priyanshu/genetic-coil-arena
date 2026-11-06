# R17 step 6b: which assumption matters most?

Read-out (fixed in advance): variants that drop the share found below 40% or push wrong finds above 5%: none.

Step 3's 40 held-out missions × 2,000 episodes; every mission replanned at mission time for each variant; one assumption moved at a time from step 6's recommended system.

| Variant | Found | Change | Wrong finds |
|---|---|---|---|
| R27: both cuts at their upper bounds (88% / 2.3%) | 49.0% | -2.8 | 4.8% |
| R20b: confirmer and closer look at the hover, then operator (92% / 2.1%) | 51.9% | +0.0 | 4.7% |
| R27: the same at the VLM's lower bound (88% / 1.5%) | 54.7% | +2.8 | 4.2% |
| R27: closer look, then Qwen3-VL-8B, then operator (88% / 1.3%) | 55.3% | +3.4 | 3.8% |
| R27: Qwen3-VL-2B instead, no bar (88% / 1.0%) | 58.8% | +6.9 | 3.6% |

Reproduce: `python scripts/r17h_sensitivity.py --from runs:r17b-train-cond-*`.
