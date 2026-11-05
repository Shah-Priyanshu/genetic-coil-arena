# R24b: the MaixCAM scout brain with the heat conv calibrated to its clamp

**PASS.** Per seed the compiled model's threshold is set on VisDrone val to float's false detections per tile at 0.5; test-dev bars: people found at most 2 points below float's at 0.5, false detections per tile within 10%.

| Seed | int8 threshold | Recall float / int8 | False per tile float / int8 | 0-8 px | 8-16 px | 16-32 px | 32+ px | cvimodel | NPU memory | Passes |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.484 | 26.1% / 25.3% | 0.259 / 0.247 (-5%) | 11.8 / 11.0 | 28.9 / 27.8 | 46.6 / 45.9 | 65.5 / 65.0 | 521 KB | 2.47 MB | yes |
| 1 | 0.484 | 26.3% / 26.0% | 0.281 / 0.282 (+0%) | 11.7 / 11.4 | 29.5 / 28.9 | 46.9 / 46.9 | 64.8 / 65.2 | 521 KB | 2.47 MB | yes |

Recall by person size in px of the 448 tile. NPU memory is the ION memory `model_tool --info` reports. Distinct int8 scores between 0.45 and 0.56 on val: 14 / 14.

Reproduce: `scripts/r24_maixcam_sim.py` (WSL, tpu-mlir venv) `calib`, `compile --heat-threshold 4`, `simulate`, then `python scripts/r24_maixcam.py report-b`.
