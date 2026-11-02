# R15d: training on every empty tile against false detections

**FAIL.** At matched recall (28.8% of 8-16 px people), the brain trained on every empty tile raises 0.72× the false detections per tile of R15b's brain (bar: at most 0.5×).

| Brain | Seed | Threshold | False detections per tile | Recall 0-8 | Recall 8-16 | Recall 16-32 | Recall 32+ |
|---|---|---|---|---|---|---|---|
| R15b | 0 | 0.502 | 0.0971 | 10.5% | 28.8% | 44.9% | 62.1% |
| R15b | 1 | 0.496 | 0.0975 | 10.6% | 28.8% | 45.1% | 61.4% |
| R15b | 2 | 0.501 | 0.0962 | 10.4% | 28.8% | 45.0% | 60.9% |
| all empty tiles | 0 | 0.485 | 0.0706 | 10.3% | 28.8% | 47.1% | 61.3% |
| all empty tiles | 1 | 0.488 | 0.0691 | 10.3% | 28.8% | 46.3% | 62.9% |

VisDrone2019-DET test-dev tiles (224, native and half scale), centre hits; width 0.5, fine-tuned from COCO, 20 epochs; the only change is the share of empty training tiles (20% → 100%).

Reproduce: `python scripts/r15d_negatives.py --from runs:r15b2-train-ft-a05 runs:r15d-train-neg`.
