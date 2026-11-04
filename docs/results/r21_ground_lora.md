# R21: the ground station's language model with its weights edited

**PASS.** Qwen2.5-Coder-7B with QLoRA, on G14b's sealed set (150 requests): exact 91.1% [83, 95] (bar >= 90%), wrong commands 2.0% [1, 6] (bar <= 3%, upper <= 6%).

Editing: +10.0 points of exact match over the same 7B as is (McNemar p = 0.0352; meets the >= 10-point bar).

Against the deployed 14B on the same items (no bar): exact match 7 items only the tuned 7B gets, 5 only the 14B (p = 0.77); right outcome overall (exact, asked, refused as expected) 19 against 8 (p = 0.052).

Training requests: 1250 written by five writers from the label guide alone; 68 dropped by the label check, 212 as too close to a dev or sealed text; 997 kept.

| Epoch | Train loss | Dev exact | Dev wrong commands | Dev correct |
|---|---|---|---|---|
| 1 | 0.0646 | 35/36 | 0 | 59/60 |
| 2 | 0.0080 | 35/36 | 0 | 59/60 |
| 3 | 0.0020 | 35/36 | 0 | 59/60 |

Chosen on dev: epoch 1.

| System | Split | Exact (ok items) | Asked when unclear | Refused when should | Wrong commands | Median latency |
|---|---|---|---|---|---|---|
| Qwen2.5-Coder-7B as is | dev | 88.9% [75, 96] | 83.3% [55, 95] | 75.0% [47, 91] | 3.3% [1, 11] | 1.32 s |
| Qwen2.5-Coder-7B as is | sealed-b | 81.1% [72, 88] | 86.7% [70, 95] | 83.3% [66, 93] | 3.3% [1, 8] | 2.54 s |
| Qwen2.5-Coder-7B + QLoRA | sealed-b | 91.1% [83, 95] | 100.0% [89, 100] | 86.7% [70, 95] | 2.0% [1, 6] | 2.88 s |

G14b reference (qwen2.5-coder:14b as is, Ollama structured output, same sealed set): exact 88.9%, wrong commands 2.7%.

Sealed misses of the tuned model:

| Id | Request | Expected | Got |
|---|---|---|---|
| b006 | Marta wants the swarm to ring the old barn tonight, with 3 m spacing between the drones please | refuse | {'behavior': 'formation', 'shape': 'circle', 'spacing_m': 3.0} |
| b009 | watch the launch pad, 30 feet out | {'behavior': 'perimeter', 'area': 'home', 'radius_m': 9.14} | asked |
| b013 | Birds mode. separation four, alignment half. | {'behavior': 'boids', 'separation': 4, 'alignment': 0.5} | refused |
| b015 | Search the schoolyard | {'behavior': 'search_grid', 'area': 'school'} | asked |
| b019 | circle the barn while doing a rainbow show | refuse | failed |
| b045 | make a circle | {'behavior': 'formation', 'shape': 'circle'} | asked |
| b050 | search the barn but don't move | refuse | {'behavior': 'search_grid', 'area': 'barn', 'alt_m': 5.0} |
| b063 | search the orchard then return home | refuse | {'behavior': 'search_grid', 'area': 'orchard', 'alt_m': 5.0} |
| b070 | circle the motorway | {'behavior': 'perimeter', 'area': 'highway'} | asked |
| b073 | serach the river | {'behavior': 'search_grid', 'area': 'river bank'} | asked |
| b143 | Search the parking lot | {'behavior': 'search_grid', 'area': 'car park'} | asked |
| b145 | search the forest at ten metres | {'behavior': 'search_grid', 'area': 'east woods', 'alt_m': 10} | asked |

Reproduce: `python scripts/r21_ground_lora.py report`.
