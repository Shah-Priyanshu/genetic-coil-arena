# R21c: the ground station's QLoRA edit on a fresh sealed set (G14c)

**NOT PASSED** (G17's bars for each adapter: exact >= 90% of the ok items, wrong commands <= 3% with the Wilson upper bound <= 6%; KILL if exact < 80% or wrong commands > 10%). G14c: 150 requests (90 / 30 / 30) written by a separate agent from G14b's label guide alone, after both adapters were trained; run once per system.

| System | Exact (ok items) | Wrong commands | Asked when unclear | Refused when it should | Correct overall | Passes |
|---|---|---|---|---|---|---|
| seed 0 (R21) | 94.4% [88, 98] | 4.0% [2, 8] | 96.7% [83, 99] | 70.0% [52, 83] | 90.0% [84, 94] | no |
| seed 1 (R21b) | 98.9% [94, 100] | 0.7% [0, 4] | 90.0% [74, 97] | 96.7% [83, 99] | 96.7% [92, 99] | yes |
| 7B as is | 80.0% [71, 87] | 2.7% [1, 7] | 70.0% [52, 83] | 76.7% [59, 88] | 77.3% [70, 83] | (no bar) |
| 14B as is (deployed before R21) | 90.0% [82, 95] | 4.0% [2, 8] | 66.7% [49, 81] | 86.7% [70, 95] | 84.7% [78, 90] | (no bar) |

Paired exact matches on the ok items (exact McNemar):

- seed 1 (R21b) only 5, seed 0 (R21) only 1 (p = 0.219).
- seed 1 (R21b) only 18, 7B as is only 1 (p = 7.63e-05).
- seed 1 (R21b) only 9, 14B as is (deployed before R21) only 1 (p = 0.0215).

Misses by kind (expected -> got):

- seed 0 (R21): clarify -> wrong command 1, ok -> asked 5, refuse -> asked 4, refuse -> wrong command 5
- seed 1 (R21b): clarify -> refused 2, clarify -> wrong command 1, ok -> refused 1, refuse -> asked 1
- 7B as is: clarify -> refused 8, clarify -> wrong command 1, ok -> asked 7, ok -> failed 2, ok -> refused 9, refuse -> asked 4, refuse -> wrong command 3
- 14B as is (deployed before R21): clarify -> refused 8, clarify -> wrong command 2, ok -> asked 3, ok -> refused 5, ok -> wrong command 1, refuse -> asked 1, refuse -> wrong command 3

Request texts are not shown, so the set can test the next change. Reproduce: `python scripts/r21_ground_lora.py report-c`.
