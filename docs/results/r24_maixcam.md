# R24: the 448-px scout brain through MaixCAM's toolchain (tpu-mlir, cv181x)

**NOT PASSED.** Per seed the compiled model's threshold is set on VisDrone val to float's false detections per tile at 0.5; test-dev bars: people found at most 2 points below float's at 0.5, false detections per tile within 10%.

| Seed | int8 threshold | Recall float / int8 | False per tile float / int8 | 0-8 px | 8-16 px | 16-32 px | 32+ px | cvimodel | NPU memory | Passes |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.500 | 26.1% / 22.7% | 0.259 / 0.197 (-24%) | 11.8 / 9.7 | 28.9 / 25.1 | 46.6 / 40.3 | 65.5 / 60.8 | 521 KB | 2.47 MB | no |
| 1 | 0.500 | 26.3% / 23.2% | 0.281 / 0.221 (-21%) | 11.7 / 10.1 | 29.5 / 25.7 | 46.9 / 41.3 | 64.8 / 60.9 | 521 KB | 2.47 MB | no |

Recall by person size in px of the 448 tile. NPU memory is the ION memory `model_tool --info` reports.

Reproduce: `scripts/r24_maixcam_sim.py` (WSL, tpu-mlir venv) `calib`, `compile`, `simulate`, then `python scripts/r24_maixcam.py report`.
