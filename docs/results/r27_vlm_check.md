# R27: a ground-station vision-language model as the final check

**NOT PASSED.** Qwen3-VL-8B-Instruct (4-bit), variant `marked` chosen on dev, at the dev threshold that keeps 95% of the dev people: rejects 39.3% [28, 52] of the test clear non-persons and keeps 95.3% [94, 96] of the test people (bars: rejects >= 50% with the lower bound >= 30%, keeps >= 93% and each group >= 90%; KILL below 25% rejected).

Test: R25c's blind-labelled false alarms of R25b's four chains plus 100 VisDrone people passed per chain; 398 spots near a dev spot dropped. Counts: {'person': 1420, 'not_person': 56, 'ambiguous': 126}.

| Model | Split | Variant | Clear non-persons rejected | Unclear rejected | People kept (blind-labelled) | People kept (VisDrone) | AUC people vs clear |
|---|---|---|---|---|---|---|---|
| 8b | dev | plain | 24.5% [15, 38] | 2.4% [1, 7] | 95.6% [94, 97] | 94.5% [90, 97] | 0.750 |
| 8b | dev | marked | 52.8% [40, 66] | 32.5% [25, 41] | 96.1% [94, 97] | 94.5% [90, 97] | 0.890 |
| 2b | dev | plain | 35.8% [24, 49] | 6.5% [3, 12] | 94.5% [92, 96] | 96.5% [93, 98] | 0.740 |
| 2b | dev | marked | 49.1% [36, 62] | 22.8% [16, 31] | 94.7% [93, 96] | 97.0% [94, 99] | 0.851 |
| 8b | test | marked | 39.3% [28, 52] | 27.0% [20, 35] | 95.1% [94, 96] | 95.7% [93, 97] | 0.828 |
| 2b | test | marked | 53.6% [41, 66] | 22.2% [16, 30] | 94.6% [93, 96] | 96.8% [94, 98] | 0.847 |

Reproduce: `python scripts/r27_vlm_check.py dev`, `test`, `report`.
