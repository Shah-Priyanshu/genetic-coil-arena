# R21b: the ground station's QLoRA edit with a second training seed

**PASS** (G17's bars: exact >= 90% of the ok items, wrong commands <= 3% with the Wilson upper bound <= 6%). G14b's sealed set, run once per adapter; same training data, recipe and dev epoch rule, only the seed differs.

| Adapter | Exact (ok items) | Wrong commands | Asked when unclear | Refused when it should | Correct overall |
|---|---|---|---|---|---|
| seed 0 (R21) | 91.1% [83, 95] | 2.0% [1, 6] | 100.0% [89, 100] | 86.7% [70, 95] | 92.0% [87, 95] |
| seed 1 | 94.4% [88, 98] | 0.0% [0, 2] | 100.0% [89, 100] | 100.0% [89, 100] | 96.7% [92, 99] |

Seed 1 chose epoch 2 on dev. Paired exact matches: seed 1 only 6, seed 0 only 3 (exact McNemar p = 0.508).

Reproduce: `python scripts/r21_ground_lora.py report-b`.
