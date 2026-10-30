# R15c probe: the scout brain at a 448×448 input

No pre-registered bars. The question: is recall at a given person size in pixels the same at this input as at 224? If it is (within about 5 points per size bin at threshold 0.5), pixels on the person summarise a camera and its resolution; compare the operating-point table with R15b's.

VisDrone2019-DET test-dev: 1610 drone frames cut into 448×448 tiles at native and half resolution, scored once after all training. Intervals: 95% paired bootstrap over frames. Centre AP counts a detection when its centre falls inside the person's box (grown to at least 8×8 px), which is what OpenSwarm's DETECTION packet carries. Several seeds are shown as mean (seed values).

| Brain | Width | int8 weights | AP50 | Centre AP | Present AUC | COCO val AP50 (ground photos) |
|---|---|---|---|---|---|---|
| zero-shot | 0.25 | - | 0.023 | 0.075 | 0.667 | 0.241 |
| zero-shot | 0.5 | - | 0.035 | 0.085 | 0.662 | 0.281 |
| zero-shot | 0.75 | - | 0.046 | 0.095 | 0.644 | 0.304 |
| fine-tuned | 0.5 | 434 KB | 0.188 | 0.369 | 0.882 | 0.103 |

Centre AP by person size (square root of box area, in pixels of the 448 tile; persons per bin in the header):

| Brain | Width | 0-8 px (41,893) | 8-16 px (29,130) | 16-32 px (14,995) | 32+ px (5,422) |
|---|---|---|---|---|---|
| zero-shot | 0.25 | 0.016 | 0.097 | 0.202 | 0.243 |
| zero-shot | 0.5 | 0.015 | 0.111 | 0.224 | 0.302 |
| zero-shot | 0.75 | 0.018 | 0.116 | 0.237 | 0.340 |
| fine-tuned | 0.5 | 0.179 | 0.405 | 0.637 | 0.839 |

AP50 by person size (square root of box area, in pixels of the 448 tile; persons per bin in the header):

| Brain | Width | 0-8 px (41,893) | 8-16 px (29,130) | 16-32 px (14,995) | 32+ px (5,422) |
|---|---|---|---|---|---|
| zero-shot | 0.25 | 0.001 | 0.017 | 0.063 | 0.160 |
| zero-shot | 0.5 | 0.001 | 0.028 | 0.100 | 0.238 |
| zero-shot | 0.75 | 0.001 | 0.037 | 0.134 | 0.274 |
| fine-tuned | 0.5 | 0.034 | 0.201 | 0.455 | 0.660 |

By tile scale (1 = native VisDrone pixels; 0.5 = frame halved, people as from twice the height):

| Brain | Width | AP50, scale 1 | AP50, scale 0.5 | Centre AP, scale 1 | Centre AP, scale 0.5 |
|---|---|---|---|---|---|
| zero-shot | 0.25 | 0.033 | 0.015 | 0.090 | 0.065 |
| zero-shot | 0.5 | 0.044 | 0.028 | 0.096 | 0.082 |
| zero-shot | 0.75 | 0.057 | 0.038 | 0.107 | 0.094 |
| fine-tuned | 0.5 | 0.250 | 0.135 | 0.437 | 0.309 |

What a scout sees at a fixed confidence threshold (centre hits, scales pooled): recall by person size, false detections per tile, and the share of empty tiles that raise an alarm. 0.7 is OpenSwarm's person-follow threshold.

| Brain | Width | Threshold | recall 0-8 px | recall 8-16 px | recall 16-32 px | recall 32+ px | False detections per tile | Empty tiles alarmed |
|---|---|---|---|---|---|---|---|---|
| zero-shot | 0.25 | 0.5 | 0.006 | 0.027 | 0.068 | 0.198 | 0.099 | 0.063 |
| zero-shot | 0.25 | 0.7 | 0.000 | 0.001 | 0.004 | 0.045 | 0.002 | 0.001 |
| zero-shot | 0.5 | 0.5 | 0.018 | 0.072 | 0.145 | 0.346 | 0.422 | 0.231 |
| zero-shot | 0.5 | 0.7 | 0.002 | 0.008 | 0.017 | 0.143 | 0.022 | 0.016 |
| zero-shot | 0.75 | 0.5 | 0.025 | 0.106 | 0.243 | 0.471 | 0.831 | 0.451 |
| zero-shot | 0.75 | 0.7 | 0.006 | 0.024 | 0.062 | 0.270 | 0.093 | 0.075 |
| fine-tuned | 0.5 | 0.5 | 0.118 | 0.289 | 0.466 | 0.655 | 0.260 | 0.078 |
| fine-tuned | 0.5 | 0.7 | 0.022 | 0.061 | 0.081 | 0.290 | 0.008 | 0.002 |

Training: 20 epochs on 59,146 VisDrone train tiles (every tile with a person, 20% of the others), batch 128, AdamW one-cycle, peak learning rate 0.001, the same settings for fine-tuned and scratch. Zero-shot = R15's COCO models (trained at 160, run at 224).

Reproduce: `python scripts/r15b_aerial_scout.py prepare`, `train --init coco` and `--init scratch`, then `report --from runs:r15b2-train-*`.
