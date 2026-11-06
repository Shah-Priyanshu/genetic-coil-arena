# R21d: the station with code refusals (v6) on G14c

**NOT PASSED** (G17's bars for each adapter: exact >= 90% of the ok items, wrong commands <= 3% with the Wilson upper bound <= 6%; KILL if exact < 80% or wrong commands > 10%). G14c: 150 requests (90 / 30 / 30) written by a separate agent from G14b's label guide alone, after both adapters were trained; run once per system. Station: v6 (v5 plus `ground.code_refusals`, written on training and dev texts only).

| System | Exact (ok items) | Wrong commands | Asked when unclear | Refused when it should | Correct overall | Passes |
|---|---|---|---|---|---|---|
| seed 0 (R21) | 93.3% [86, 97] | 3.3% [1, 8] | 96.7% [83, 99] | 73.3% [56, 86] | 90.0% [84, 94] | no |
| seed 1 (R21b) | 97.8% [92, 99] | 0.7% [0, 4] | 90.0% [74, 97] | 96.7% [83, 99] | 96.0% [92, 98] | yes |
| 7B as is | 78.9% [69, 86] | 2.0% [1, 6] | 70.0% [52, 83] | 80.0% [63, 90] | 77.3% [70, 83] | (no bar) |

Paired exact matches on the ok items (exact McNemar):

- seed 1 (R21b) only 5, seed 0 (R21) only 1 (p = 0.219).
- seed 1 (R21b) only 18, 7B as is only 1 (p = 7.63e-05).

Against the same model in v5 (R21c), right outcome over all 150 (exact McNemar):

- seed 0 (R21): right only in v6 1, only in v5 1 (p = 1).
- seed 1 (R21b): right only in v6 0, only in v5 1 (p = 1).
- 7B as is: right only in v6 1, only in v5 1 (p = 1).

Misses by kind (expected -> got):

- seed 0 (R21): clarify -> wrong command 1, ok -> asked 5, ok -> refused 1, refuse -> asked 4, refuse -> wrong command 4
- seed 1 (R21b): clarify -> refused 2, clarify -> wrong command 1, ok -> refused 2, refuse -> asked 1
- 7B as is: clarify -> refused 8, clarify -> wrong command 1, ok -> asked 7, ok -> failed 2, ok -> refused 10, refuse -> asked 4, refuse -> wrong command 2

Request texts are not shown, so the set can test the next change. Reproduce: `python scripts/r21_ground_lora.py report-d`.
