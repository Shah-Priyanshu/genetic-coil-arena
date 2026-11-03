# R19: an open-weight confirmer for the scout's detections (goal G15)

**Verdict: PASS**; editing the weights misses its bar: the chosen model passes 0.56× [0.52, 0.60] of the test-dev false alarms the same model passes unedited (bar ≤ 0.5×, upper bound < 1)

Candidates are R15b's width-0.5 brain's detections at confidence ≥ 0.5 on VisDrone-DET frames at native scale; each model sees the 144-px view around the detection. Every threshold keeps 95% of the people in the val selection set (7638 candidates, 77% people). Test-dev: 11639 candidates from 1336 frames, 70% people (VisDrone labels). Chosen model (lowest val pass among the edited ones): `dino-full-e10s336_s2`.

## Val (selection) and test-dev, VisDrone labels

| Model | Seed | Val AUC | Val false alarms passed | Test AUC | Test people kept | Test false alarms passed |
|---|---|---|---|---|---|---|
| DINOv2-small, fine-tuned (e10) | 0 | 0.881 | 36.5% | – | – | – |
| DINOv2-small, fine-tuned (e10s336) | 0 | 0.887 | 34.5% | 0.913 | 91.4% [91, 92] | 24.3% [23, 26] |
| DINOv2-small, fine-tuned (e10s336) | 1 | 0.887 | 36.1% | 0.910 | 91.3% [91, 92] | 24.2% [23, 26] |
| DINOv2-small, fine-tuned (e10s336) | 2 | 0.890 | 33.9% | 0.911 | 90.9% [90, 91] | 23.7% [22, 25] |
| DINOv2-small, fine-tuned | 0 | 0.878 | 38.4% | – | – | – |
| DINOv2-small, fine-tuned | 1 | 0.882 | 37.5% | – | – | – |
| DINOv2-small, fine-tuned | 2 | 0.881 | 38.7% | – | – | – |
| DINOv2-small, weights kept (linear probe) (e10s336) | 0 | 0.725 | 65.3% | 0.805 | 89.5% [89, 90] | 42.3% [41, 44] |
| DINOv2-small, weights kept (linear probe) | 0 | 0.715 | 68.0% | – | – | – |
| Qwen3-VL-2B, LoRA-tuned | 0 | 0.866 | 38.0% | 0.892 | 90.5% [90, 91] | 25.8% [24, 27] |
| Qwen3-VL-2B, as is (asked) | 0 | 0.675 | 72.4% | 0.726 | 93.0% [92, 94] | 58.8% [57, 60] |

Edited against the same model unedited (test-dev false alarms passed, frame-bootstrap 95% interval):

- `dino-full-e10s336_s0` vs `dino-probe-e10s336_s0`: 0.57× [0.54, 0.61]
- `dino-full-e10s336_s1` vs `dino-probe-e10s336_s0`: 0.57× [0.53, 0.61]
- `dino-full-e10s336_s2` vs `dino-probe-e10s336_s0`: 0.56× [0.52, 0.60]
- `qwen-lora_s0` vs `qwen-zero_s0`: 0.44× [0.41, 0.47]

## The 400 operator spots (R15e and R15f, val)

Claude's visual checks: 200 matched people, 53 unlabelled people, 81 clear non-persons, 66 ambiguous. "Chained" = the model passes the spot and the operator said person.

| Who | Matched people kept | Clear non-persons passed | Ambiguous passed | Unlabelled people kept |
|---|---|---|---|---|
| Operator alone | 97.5% [94, 99] | 16.0% [10, 26] | 51.5% [40, 63] | 67.9% [55, 79] |
| DINOv2-small, fine-tuned (e10s336) (seed 0) | 95.5% [92, 98] | 9.9% [5, 18] | 22.7% [14, 34] | 43.4% [31, 57] |
| DINOv2-small, fine-tuned (e10s336) + operator (seed 0) | 94.0% [90, 97] | 4.9% [2, 12] | 18.2% [11, 29] | 34.0% [23, 47] |
| DINOv2-small, fine-tuned (e10s336) (seed 1) | 95.5% [92, 98] | 7.4% [3, 15] | 22.7% [14, 34] | 47.2% [34, 60] |
| DINOv2-small, fine-tuned (e10s336) + operator (seed 1) | 94.0% [90, 97] | 4.9% [2, 12] | 18.2% [11, 29] | 39.6% [28, 53] |
| DINOv2-small, fine-tuned (e10s336) (seed 2) | 93.0% [89, 96] | 6.2% [3, 14] | 18.2% [11, 29] | 45.3% [33, 59] |
| DINOv2-small, fine-tuned (e10s336) + operator (seed 2) | 92.0% [87, 95] | 4.9% [2, 12] | 15.2% [8, 26] | 35.8% [24, 49] |
| DINOv2-small, weights kept (linear probe) (e10s336) (seed 0) | 95.5% [92, 98] | 27.2% [19, 38] | 68.2% [56, 78] | 79.2% [67, 88] |
| DINOv2-small, weights kept (linear probe) (e10s336) + operator (seed 0) | 93.0% [89, 96] | 6.2% [3, 14] | 36.4% [26, 48] | 60.4% [47, 72] |
| Qwen3-VL-2B, LoRA-tuned (seed 0) | 94.5% [90, 97] | 2.5% [1, 9] | 21.2% [13, 33] | 49.1% [36, 62] |
| Qwen3-VL-2B, LoRA-tuned + operator (seed 0) | 93.5% [89, 96] | 2.5% [1, 9] | 18.2% [11, 29] | 37.7% [26, 51] |
| Qwen3-VL-2B, as is (asked) (seed 0) | 95.0% [91, 97] | 40.7% [31, 52] | 75.8% [64, 84] | 92.5% [82, 97] |
| Qwen3-VL-2B, as is (asked) + operator (seed 0) | 93.0% [89, 96] | 11.1% [6, 20] | 43.9% [33, 56] | 62.3% [49, 74] |

### DINOv2-small, fine-tuned (e10s336) (seed 0) on test-dev by detection size

| Size (px) | People kept | False alarms passed |
|---|---|---|
| 0-8 | 81.4% [78, 84] | 14.6% [12, 17] |
| 8-16 | 89.7% [89, 91] | 22.6% [21, 25] |
| 16-32 | 94.8% [94, 96] | 28.4% [25, 32] |
| 32+ | 93.2% [92, 94] | 40.6% [36, 46] |

### DINOv2-small, fine-tuned (e10s336) (seed 1) on test-dev by detection size

| Size (px) | People kept | False alarms passed |
|---|---|---|
| 0-8 | 80.5% [77, 83] | 13.0% [11, 16] |
| 8-16 | 89.4% [88, 90] | 22.5% [20, 25] |
| 16-32 | 95.2% [94, 96] | 29.3% [26, 32] |
| 32+ | 93.2% [92, 94] | 40.9% [36, 46] |

### DINOv2-small, fine-tuned (e10s336) (seed 2) on test-dev by detection size

| Size (px) | People kept | False alarms passed |
|---|---|---|
| 0-8 | 80.6% [77, 83] | 12.5% [10, 15] |
| 8-16 | 88.1% [87, 89] | 21.9% [20, 24] |
| 16-32 | 95.1% [94, 96] | 29.1% [26, 32] |
| 32+ | 94.1% [93, 95] | 39.8% [35, 45] |

### DINOv2-small, weights kept (linear probe) (e10s336) (seed 0) on test-dev by detection size

| Size (px) | People kept | False alarms passed |
|---|---|---|
| 0-8 | 74.0% [71, 77] | 25.6% [23, 29] |
| 8-16 | 85.2% [84, 86] | 41.8% [39, 44] |
| 16-32 | 95.8% [95, 96] | 50.5% [47, 54] |
| 32+ | 95.2% [94, 96] | 57.3% [52, 62] |

### Qwen3-VL-2B, LoRA-tuned (seed 0) on test-dev by detection size

| Size (px) | People kept | False alarms passed |
|---|---|---|
| 0-8 | 73.9% [70, 77] | 12.0% [10, 15] |
| 8-16 | 87.3% [86, 88] | 22.5% [21, 25] |
| 16-32 | 96.2% [95, 97] | 34.4% [31, 38] |
| 32+ | 94.9% [94, 96] | 46.0% [41, 51] |

### Qwen3-VL-2B, as is (asked) (seed 0) on test-dev by detection size

| Size (px) | People kept | False alarms passed |
|---|---|---|
| 0-8 | 89.9% [87, 92] | 46.4% [43, 50] |
| 8-16 | 96.2% [95, 97] | 62.5% [60, 65] |
| 16-32 | 94.1% [93, 95] | 61.5% [58, 65] |
| 32+ | 84.2% [82, 86] | 59.9% [55, 65] |
