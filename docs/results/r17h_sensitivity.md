# R17 step 6b: which assumption matters most?

Read-out (fixed in advance): variants that drop the share found below 40% or push wrong finds above 5%: 1 frame per second of video, filter keeps 80% of detections, filter keeps 60% of detections, false alarms cut 5x (not 10x), operator accepts 5% of empty views, operator accepts 10% of empty views, person smaller (0.28 m).

Step 3's 40 held-out missions × 2,000 episodes; every mission replanned at mission time for each variant; one assumption moved at a time from step 6's recommended system.

| Variant | Found | Change | Wrong finds |
|---|---|---|---|
| operator accepts 10% of empty views | 32.3% | -20.0 | 11.1% |
| 1 frame per second of video | 35.3% | -17.0 | 4.4% |
| filter keeps 60% of detections | 38.8% | -13.5 | 5.2% |
| false alarms cut 5x (not 10x) | 39.7% | -12.6 | 5.4% |
| operator accepts 5% of empty views | 40.1% | -12.2 | 7.1% |
| person smaller (0.28 m) | 44.2% | -8.1 | 5.1% |
| filter keeps 80% of detections | 46.1% | -6.2 | 5.1% |
| operator sees a person 50% of looks | 47.3% | -5.0 | 4.4% |
| 2 frames per second of video | 49.6% | -2.7 | 4.9% |
| operator sees a person 70% of looks | 52.2% | -0.1 | 4.9% |
| recommended (as in step 6) | 52.3% | +0.0 | 4.8% |
| person larger (0.40 m) | 58.4% | +6.1 | 4.4% |
| false alarms cut 20x | 63.5% | +11.2 | 4.1% |

Reproduce: `python scripts/r17h_sensitivity.py --from runs:r17b-train-cond-*`.
