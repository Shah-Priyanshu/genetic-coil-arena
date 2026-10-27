# R7a step 2: an honest G3 translator, and a distilled model that fits beside the specialists

Generated 2026-10-27 05:39 on PC (Python 3.13.7). Command: `python scripts/r7a_step2.py aggregate --runs runs:r7a3-* runs:r7a2-eval-ollama --stem docs/results/r7a_step3`.

> **This is the R7a v2 distillation rerun (jobs `r7a3-*`), rendered with the step-2 report template.** The Ollama rows (14B, gemma4:e4b) and the keyword parser are step 2's runs, unchanged. Only the distilled model is new.
> - **What changed:** 11,520 paraphrases from qwen3-coder:30b (seed 1), of which 8,086 survived the 14B check and the leak filter (step 2 had about 2,000). The run also swept LoRA rank and data size (four runs, rank 32 or 64, 12k or 24k records, 50% or 80% paraphrases) and selected on dev.
> - **What it showed:**
>   - Dev rose to 95.8% for `r32-n12k-p80`, the run the eval job selected. The sweep table marks two runs "(selected)"; that is a rendering bug, and the eval log names `r32-n12k-p80` only.
>   - Its single sealed evaluation scored 82.8% [75.8, 88.0], the same as step 2.
>   - The failures are mostly invented constraints: valid requests refused as "unsatisfiable", or "there is no green snake". It also flips caution polarity ("don't take chances" read as low caution).
>   - Refusal precision is 60.7%.
>   - So paraphrase volume is not what limits the 1.7B model. G3 stays passed by the prompted 14B, and the small-translator line is closed for now (`docs/goals.md`).

## What changed since step 1

- **Sealed test set.** 174 hand-written items (`experiments/data/r7a_sealed_test.jsonl`), written from the specs before any step-2 prompt or model work, none labelled by a model. SHA-256 `ca6ff07b5fb66ef21c288ba9a82d5e7763992fe2142b1a582baf597e542b5e01`, asserted by `tests/test_r7a_sealed.py` and by every script that loads it. Tuning (prompts, training data, checkpoint choice) used the step-1 set and a separate dev set only; each translator was run on the sealed set once.
- **Refusal policy** (`translator/policy.py`). Parse first, validate after: the model extracts into a relaxed schema; code fills documented defaults, asks for a missing mandatory choice, and validates. Refusal only for out-of-domain, contradictory or unsatisfiable requests (a value the schema forbids is sent back once, then refused). A new `clarify` outcome returns a question for an in-domain but ambiguous request; it is neither an exact match nor a refusal and is reported separately.
- **Distilled translator.** A LoRA on Qwen3-1.7B that writes the compact policy reply (params that differ from the default; a refusal; or a question), with the same schema check and one retry at inference. Training data: templates plus paraphrases from qwen3-coder:30b, each kept only if qwen2.5-coder:14b recovers the gold outcome from it, plus retry-format examples; anything whose phrasing frame overlaps the sealed, dev or step-1 sets (4-gram containment >= 0.5) is dropped, and the training job refuses to start if any survives.

Sealed set: 174 items, 145 with a gold spec, the rest gold refusals or gold clarifications; squad and plant.

## Verdict

| Translator | G3 verdict | Exact match (sealed) | Round trip | Warm p95 | Inference memory |
|---|---|---|---|---|---|
| qwen2.5-coder:14b (refusal policy) | **PASS** | 95.9% (91%-98%, n=145) | 100.0% | 1.39 s | 9.0 GB |
| Qwen3-1.7B + LoRA (distilled) | **KILL criterion hit** | 82.8% (76%-88%, n=145) | 90.7% | 5.38 s | 3.37 GB |
| gemma4:e4b (refusal policy) | **NOT PASSED (no kill criterion hit)** | 92.4% (87%-96%, n=145) | 100.0% | 0.84 s | 4.3 GB |
| keyword/regex parser | **KILL criterion hit** | 36.6% (29%-45%, n=145) | 62.4% | n/a | n/a |

### qwen2.5-coder:14b (refusal policy)

| Criterion | Target | Measured | Result |
|---|---|---|---|
| G3 exact match, sealed valid items | >= 95% (kill < 85%) | 95.9% (n=145, 95% CI 91.3%-98.1%) | PASS |
| G3 task score within CI of gold-spec run | paired diff >= 0 or inside its CI, both domains | squad: NL 8.75 vs gold 8.83, diff -0.08 ± 0.14; plant: NL 0.60 vs gold 0.60, diff +0.00 ± 0.00 | PASS |
| G3 round trip (reverse summary re-translated) | >= 90% (kill < 70%) | 100.0% (n=143, 95% CI 97.4%-100.0%) | PASS |
| G6 summary field fidelity | >= 90% (kill < 75%) | 100.0% of facts (all facts in 100.0% of summaries) | PASS |
| G3 warm translator p95 | < 2 s | 1.39 s | PASS |
| G3 fits in 12 GB | <= 12 GB | 9.0 GB peak | PASS |

### Qwen3-1.7B + LoRA (distilled)

| Criterion | Target | Measured | Result |
|---|---|---|---|
| G3 exact match, sealed valid items | >= 95% (kill < 85%) | 82.8% (n=145, 95% CI 75.8%-88.0%) | KILL |
| G3 task score within CI of gold-spec run | paired diff >= 0 or inside its CI, both domains | squad: NL 7.78 vs gold 8.83, diff -1.05 ± 0.26; plant: NL 0.56 vs gold 0.60, diff -0.05 ± 0.00 | not met |
| G3 round trip (reverse summary re-translated) | >= 90% (kill < 70%) | 90.7% (n=129, 95% CI 84.4%-94.6%) | PASS |
| G6 summary field fidelity | >= 90% (kill < 75%) | 94.1% of facts (all facts in 90.7% of summaries) | PASS |
| G3 warm translator p95 | < 2 s | 5.38 s | not met |
| G3 fits in 12 GB | <= 12 GB | 3.4 GB peak | PASS |
| Step-2 target: inference memory beside the specialists | <= 4 GB (torch peak allocated) | 3.37 GB peak allocated; 4.45 GB by nvidia-smi incl. CUDA context | PASS |

### gemma4:e4b (refusal policy)

| Criterion | Target | Measured | Result |
|---|---|---|---|
| G3 exact match, sealed valid items | >= 95% (kill < 85%) | 92.4% (n=145, 95% CI 86.9%-95.7%) | not met |
| G3 task score within CI of gold-spec run | paired diff >= 0 or inside its CI, both domains | squad: NL 8.90 vs gold 8.83, diff +0.06 ± 0.13; plant: NL 0.55 vs gold 0.60, diff -0.06 ± 0.00 | not met |
| G3 round trip (reverse summary re-translated) | >= 90% (kill < 70%) | 100.0% (n=139, 95% CI 97.3%-100.0%) | PASS |
| G6 summary field fidelity | >= 90% (kill < 75%) | 100.0% of facts (all facts in 100.0% of summaries) | PASS |
| G3 warm translator p95 | < 2 s | 0.84 s | PASS |
| G3 fits in 12 GB | <= 12 GB | 4.3 GB peak | PASS |

### keyword/regex parser

| Criterion | Target | Measured | Result |
|---|---|---|---|
| G3 exact match, sealed valid items | >= 95% (kill < 85%) | 36.6% (n=145, 95% CI 29.2%-44.6%) | KILL |
| G3 task score within CI of gold-spec run | paired diff >= 0 or inside its CI, both domains | squad: NL 9.90 vs gold 8.83, diff +1.06 ± 0.63; plant: NL 0.58 vs gold 0.60, diff -0.02 ± 0.00 | not met |
| G3 round trip (reverse summary re-translated) | >= 90% (kill < 70%) | 62.4% (n=141, 95% CI 54.2%-70.0%) | KILL |
| G6 summary field fidelity | >= 90% (kill < 75%) | 100.0% of facts (all facts in 100.0% of summaries) | PASS |

## GoalSpec accuracy on the sealed set

Exact match is over the items with a gold spec (params equal after sorting unordered arrays; 95% Wilson interval). A clarifying question or a refusal on such an item counts as a miss.

| Translator | All | Squad | Plant |
|---|---|---|---|
| qwen2.5-coder:14b (refusal policy) | 95.9% (91%-98%, n=145) | 93.2% (85%-97%, n=73) | 98.6% (93%-100%, n=72) |
| Qwen3-1.7B + LoRA (distilled) | 82.8% (76%-88%, n=145) | 76.7% (66%-85%, n=73) | 88.9% (80%-94%, n=72) |
| gemma4:e4b (refusal policy) | 92.4% (87%-96%, n=145) | 95.9% (89%-99%, n=73) | 88.9% (80%-94%, n=72) |
| keyword/regex parser | 36.6% (29%-45%, n=145) | 35.6% (26%-47%, n=73) | 37.5% (27%-49%, n=72) |

Per-field accuracy:

| Translator | active_nodes | byte_budget | caution | eaters | food_target | step_limit | target_far |
|---|---|---|---|---|---|---|---|
| qwen2.5-coder:14b (refusal policy) | 100.0% | 100.0% | 93.2% | 97.3% | 97.3% | 97.3% | 98.6% |
| Qwen3-1.7B + LoRA (distilled) | 93.1% | 91.7% | 76.7% | 84.9% | 84.9% | 84.9% | 88.9% |
| gemma4:e4b (refusal policy) | 91.7% | 91.7% | 95.9% | 100.0% | 100.0% | 100.0% | 88.9% |
| keyword/regex parser | 88.9% | 77.8% | 68.5% | 87.7% | 74.0% | 76.7% | 51.4% |

## Refusals and clarifications

Refusal precision and recall are against the gold refusals (out of domain, contradictory, unsatisfiable). The clarify columns count questions asked back: the rate over all items, the share of valid-spec items that got a question instead of a spec (counted against exactness), and recall of the gold clarify items.

| Translator | Refusal precision | Refusal recall | TP / FP / FN | Clarify rate | Clarify on valid-spec items | Clarify recall | Statuses |
|---|---|---|---|---|---|---|---|
| qwen2.5-coder:14b (refusal policy) | 95.0% | 100.0% | 19 / 1 / 0 | 6.3% (11) | 0.7% (1) | 100.0% (10/10) | {'clarify': 11, 'ok': 143, 'refused': 20} |
| Qwen3-1.7B + LoRA (distilled) | 60.7% | 89.5% | 17 / 11 / 2 | 8.6% (15) | 4.1% (6) | 90.0% (9/10) | {'clarify': 15, 'ok': 131, 'refused': 28} |
| gemma4:e4b (refusal policy) | 77.8% | 73.7% | 14 / 4 / 5 | 9.2% (16) | 1.4% (2) | 100.0% (10/10) | {'clarify': 16, 'ok': 140, 'refused': 18} |
| keyword/regex parser | 61.1% | 57.9% | 11 / 7 / 8 | 0.0% (0) | 0.0% (0) | 0.0% (0/10) | {'invalid': 11, 'ok': 156, 'refused': 7} |

## Task score driven from natural language vs the gold spec

Both runs go through the command centre on the same episodes; an item whose translation failed scores 0 on every seed. Mean over items per seed, then mean ± 95% t-interval over seeds; the last columns are the paired difference (NL minus gold) and whether it lies within its CI (or is >= 0).

| Translator | Domain | n | NL | Gold | NL - gold | Within CI |
|---|---|---|---|---|---|---|
| qwen2.5-coder:14b (refusal policy) | squad | 73 | 8.75 ± 1.22 | 8.83 ± 1.10 | -0.08 ± 0.14 | yes |
| qwen2.5-coder:14b (refusal policy) | plant | 72 | 0.60 ± 0.01 | 0.60 ± 0.01 | 0.00 ± 0.00 | yes |
| Qwen3-1.7B + LoRA (distilled) | squad | 73 | 7.78 ± 0.90 | 8.83 ± 1.10 | -1.05 ± 0.26 | no |
| Qwen3-1.7B + LoRA (distilled) | plant | 72 | 0.56 ± 0.01 | 0.60 ± 0.01 | -0.05 ± 0.00 | no |
| gemma4:e4b (refusal policy) | squad | 73 | 8.90 ± 1.21 | 8.83 ± 1.10 | 0.06 ± 0.13 | yes |
| gemma4:e4b (refusal policy) | plant | 72 | 0.55 ± 0.01 | 0.60 ± 0.01 | -0.06 ± 0.00 | no |
| keyword/regex parser | squad | 73 | 9.90 ± 1.57 | 8.83 ± 1.10 | 1.06 ± 0.63 | yes |
| keyword/regex parser | plant | 72 | 0.58 ± 0.01 | 0.60 ± 0.01 | -0.02 ± 0.00 | no |

## Round trip and summary fidelity

The 14B model writes a status summary from each translator's own executed trace; the translator under test then reads the summary back. Round trip: the reconstruction equals the forward spec. Fidelity (G6): the share of required facts the summary states without contradiction (regular expressions).

| Translator | n | Round trip (95% CI) | Summary field fidelity | All facts in summary |
|---|---|---|---|---|
| qwen2.5-coder:14b (refusal policy) | 143 | 100.0% (97%-100%) | 100.0% | 100.0% |
| Qwen3-1.7B + LoRA (distilled) | 129 | 90.7% (84%-95%) | 94.1% | 90.7% |
| gemma4:e4b (refusal policy) | 139 | 100.0% (97%-100%) | 100.0% | 100.0% |
| keyword/regex parser | 141 | 62.4% (54%-70%) | 100.0% | 100.0% |

## Latency and memory

Cold is the first call (model already resident for the distilled model; for Ollama it follows an unload and includes the load). Warm is every later item, retries included.

| Translator | Cold first call | Warm item p50 / p95 | Mean attempts | Memory |
|---|---|---|---|---|
| qwen2.5-coder:14b (refusal policy) | 7.38 s | 0.99 s / 1.39 s | 1.06 | 9.0 GB (nvidia-smi over baseline) |
| Qwen3-1.7B + LoRA (distilled) | 3.88 s | 3.59 s / 5.38 s | 1.04 | weights 3.33 GB, peak allocated 3.37 GB, nvidia-smi delta 4.45 GB |
| gemma4:e4b (refusal policy) | 12.17 s | 0.52 s / 0.84 s | 1.06 | 4.3 GB (nvidia-smi over baseline) |
| keyword/regex parser | - | - | - | - |

## Distilled model: rank and data-size sweep

Dev score is exact match on the valid items of the dev set plus the step-1 set (never the sealed set); the run with the best dev score was evaluated on the sealed set.

| Run | Rank | Train records (paraphrases) | Steps | Dev score | Refusal P / R (dev) | Final loss | Train time | Adapter |
|---|---|---|---|---|---|---|---|---|
| r32-n12k-p50 **(selected)** | 32 | 12000 (5760) | 2250 | 94.8% | 54.5% / 75.0% | 0.0002 | 35 min | 133 MiB |
| r32-n12k-p80 **(selected)** | 32 | 12000 (8086) | 2250 | 95.8% | 65.0% / 81.2% | 0.0003 | 35 min | 133 MiB |
| r32-n24k-p80 | 32 | 23556 (8086) | 2944 | 91.7% | 70.0% / 87.5% | 0.0009 | 45 min | 133 MiB |
| r64-n12k-p80 | 64 | 12000 (8086) | 2250 | 92.0% | 61.9% / 81.2% | 0.0001 | 34 min | 266 MiB |

## Training-data pipeline

Pool: 23556 records ({'template': 14905, 'paraphrase': 8086, 'retry': 565}; kinds {'spec': 19467, 'refuse': 2252, 'clarify': 1272, 'retry': 565}). Paraphrases: 11520 generated by qwen3-coder:30b, 8235 kept after the 14B round-trip check. Dropped by the n-gram leak filter: {'templates': 480, 'paraphrases': 128}.

| Style card | Generated | Verified | Rate |
|---|---|---|---|
| casual_chat | 1192 | 915 | 77% |
| formal_memo | 1016 | 844 | 83% |
| indirect_hint | 904 | 548 | 61% |
| jargon | 808 | 628 | 78% |
| mixed_notation | 992 | 573 | 58% |
| multi_sentence | 1024 | 660 | 64% |
| negation | 864 | 357 | 41% |
| numbers_in_words | 968 | 724 | 75% |
| polite_question | 1088 | 943 | 87% |
| self_correction | 808 | 651 | 81% |
| terse_note | 928 | 807 | 87% |
| typos | 928 | 585 | 63% |

Verification rate by kind: clarify 54% of 1152, refuse 75% of 1152, spec 73% of 9216.

## Failures on the sealed set

**qwen2.5-coder:14b (refusal policy)**: 6 of 174 items not as expected. First 12:

- `sealed-squad-009` [spec -> clarify] "Can the blue snake grab 7 food, taking it easy and safe?"  gold `{"caution": "high", "eaters": ["blue"], "food_target": 7, "step_limit": 300}`  got `null` (How many steps should the mission last?)
- `sealed-squad-017` [spec -> refused] "How about we let blue and red eat for a quarter-thousand steps, no particular goal?"  gold `{"caution": "normal", "eaters": ["red", "blue"], "food_target": 0, "step_limit": 250}`  got `null` (step_limit must be between 50 and 300)
- `sealed-squad-026` [spec -> ok] "Blue, green, and red all hunt. They should stop after 35 food. 250 steps tops. Nothing reckless."  gold `{"caution": "high", "eaters": ["red", "green", "blue"], "food_target": 35, "step_limit": 250}`  got `{"eaters": ["red", "green", "blue"], "caution": "normal", "step_limit": 250, "food_target": 35}`
- `sealed-squad-062` [spec -> ok] "Of the three snakes only blue gets to eat. 100 steps. Low risk tolerance."  gold `{"caution": "high", "eaters": ["blue"], "food_target": 0, "step_limit": 100}`  got `{"eaters": ["blue"], "caution": "low", "step_limit": 100, "food_target": 0}`
- `sealed-squad-069` [spec -> ok] "ok so red should go eat a bunch, like 25 food, but not recklessly, and stop at 220 steps"  gold `{"caution": "high", "eaters": ["red"], "food_target": 25, "step_limit": 220}`  got `{"eaters": ["red"], "caution": "normal", "step_limit": 220, "food_target": 25}`
- `sealed-plant-055` [spec -> ok] "Sensor nodes: reactor and compressor/utilities. Bytes: fifty one. False alarms: five tenths of a percent."  gold `{"active_nodes": ["reactor", "compressor_utilities"], "byte_budget": 51, "target_far": 0.005}`  got `{"byte_budget": 51, "target_far": 0.05, "active_nodes": ["reactor", "compressor_utilities"]}`

**Qwen3-1.7B + LoRA (distilled)**: 28 of 174 items not as expected. First 12:

- `sealed-squad-006` [spec -> ok] "Hey, quick one. Green is the only one who needs to eat. Keep it to a hundred steps, and don't take chances."  gold `{"caution": "high", "eaters": ["green"], "food_target": 0, "step_limit": 100}`  got `{"caution": "low", "eaters": ["green"], "step_limit": 100, "food_target": 0}`
- `sealed-squad-010` [spec -> refused] "Only the green and the blue should hunt, I want a baker's dozen eaten, 140 ticks."  gold `{"caution": "normal", "eaters": ["green", "blue"], "food_target": 13, "step_limit": 140}`  got `null` (unsatisfiable under the schema: /params/food_target: 140 is greater than the maximum of 40)
- `sealed-squad-017` [spec -> refused] "How about we let blue and red eat for a quarter-thousand steps, no particular goal?"  gold `{"caution": "normal", "eaters": ["red", "blue"], "food_target": 0, "step_limit": 250}`  got `null` (unsatisfiable under the schema: /params/step_limit: 1000 is greater than the maximum of 30)
- `sealed-squad-018` [spec -> ok] "gren and blu foraging, 120 stpes, recless"  gold `{"caution": "low", "eaters": ["green", "blue"], "food_target": 0, "step_limit": 120}`  got `{"eaters": ["green", "blue"], "step_limit": 120, "caution": "normal", "food_target": 0}`
- `sealed-squad-022` [spec -> refused] "Blue and green: eat a score of food, 150 ticks, be bold."  gold `{"caution": "low", "eaters": ["green", "blue"], "food_target": 20, "step_limit": 150}`  got `null` (unsatisfiable under the schema: /params/food_target: 150 is greater than the maximum of 40)
- `sealed-squad-024` [spec -> clarify] "is it possible to get all of them eating for 100 steps with maximum caution?"  gold `{"caution": "high", "eaters": ["red", "green", "blue"], "food_target": 0, "step_limit": 100}`  got `null` (Which snakes (red, green or blue) should forage?)
- `sealed-squad-028` [spec -> ok] "red + blue hunt for 8 food. take no risks."  gold `{"caution": "high", "eaters": ["red", "blue"], "food_target": 8, "step_limit": 300}`  got `{"caution": "low", "eaters": ["red", "blue"], "food_target": 8, "step_limit": 300}`
- `sealed-squad-030` [spec -> ok] "Let green and blue forage; I don't mind if one dies. 90 steps, 10 food."  gold `{"caution": "low", "eaters": ["green", "blue"], "food_target": 10, "step_limit": 90}`  got `{"eaters": ["green", "blue"], "food_target": 10, "step_limit": 90, "caution": "normal"}`
- `sealed-squad-035` [spec -> refused] "forage with blue, 77 steps"  gold `{"caution": "normal", "eaters": ["blue"], "food_target": 0, "step_limit": 77}`  got `null` (step limit must be between 50 and 300)
- `sealed-squad-052` [spec -> clarify] "Only red, carefully, 85 steps, 3 food."  gold `{"caution": "high", "eaters": ["red"], "food_target": 3, "step_limit": 85}`  got `null` (How much food counts as the target?)
- `sealed-squad-053` [spec -> refused] "Make the green snake eat 1 food and call it a day."  gold `{"caution": "normal", "eaters": ["green"], "food_target": 1, "step_limit": 300}`  got `null` (there is no green snake; the colours are red, green and blue)
- `sealed-squad-058` [spec -> refused] "I want red out foraging. Wait, make that green. 120 steps."  gold `{"caution": "normal", "eaters": ["green"], "food_target": 0, "step_limit": 120}`  got `null` (there is no green snake; the colours are red, green and blue)

**gemma4:e4b (refusal policy)**: 16 of 174 items not as expected. First 12:

- `sealed-squad-026` [spec -> ok] "Blue, green, and red all hunt. They should stop after 35 food. 250 steps tops. Nothing reckless."  gold `{"caution": "high", "eaters": ["red", "green", "blue"], "food_target": 35, "step_limit": 250}`  got `{"eaters": ["red", "green", "blue"], "caution": "normal", "step_limit": 250, "food_target": 35}`
- `sealed-squad-062` [spec -> ok] "Of the three snakes only blue gets to eat. 100 steps. Low risk tolerance."  gold `{"caution": "high", "eaters": ["blue"], "food_target": 0, "step_limit": 100}`  got `{"eaters": ["blue"], "caution": "low", "step_limit": 100, "food_target": 0}`
- `sealed-squad-069` [spec -> ok] "ok so red should go eat a bunch, like 25 food, but not recklessly, and stop at 220 steps"  gold `{"caution": "high", "eaters": ["red"], "food_target": 25, "step_limit": 220}`  got `{"eaters": ["red"], "caution": "normal", "step_limit": 220, "food_target": 25}`
- `sealed-squad-076` [refuse -> clarify] "Green forages for 100 steps and also for 200 steps."  gold `null`  got `null` (Which step limit should be used?)
- `sealed-squad-079` [refuse -> ok] "Send the pink and the red ones to eat, 120 steps."  gold `null`  got `{"eaters": ["red"], "caution": "normal", "step_limit": 120, "food_target": 0}`
- `sealed-plant-006` [spec -> refused] "Run diagnosis, leave the reactor out, false alarm target 0.10%."  gold `{"active_nodes": ["separator", "stripper", "compressor_utilities"], "byte_budget": 16, "target_far": 0.001}`  got `null` (target_far must be one of [0.001, 0.005, 0.01, 0.02, 0.05] and not 0.1)
- `sealed-plant-008` [spec -> ok] "The false-alarm rate should be a twentieth; budget 16 bytes."  gold `{"active_nodes": ["reactor", "separator", "stripper", "compressor_utilities"], "byte_budget": 16, "target_far": 0.05}`  got `{"byte_budget": 16, "target_far": 0.005, "active_nodes": ["reactor", "separator", "stripper", "compressor_utilities"]}`
- `sealed-plant-017` [spec -> clarify] "Use the middle byte budget and the loosest false alarm rate."  gold `{"active_nodes": ["reactor", "separator", "stripper", "compressor_utilities"], "byte_budget": 16, "target_far": 0.05}`  got `null` (Could you specify the desired byte budget (8, 16, or 51) and the false alarm rate (0.001, )
- `sealed-plant-020` [spec -> clarify] "Ninety-nine percent specificity, sixteen bytes, everything on."  gold `{"active_nodes": ["reactor", "separator", "stripper", "compressor_utilities"], "byte_budget": 16, "target_far": 0.01}`  got `null` (The target_far must be a fraction between 0 and 1, not a percentage of specificity. Could )
- `sealed-plant-021` [spec -> refused] "Switch off the stripper, keep the rest, 8 bytes, one-tenth of one percent."  gold `{"active_nodes": ["reactor", "separator", "compressor_utilities"], "byte_budget": 8, "target_far": 0.001}`  got `null` (target_far must be one of [0.001, 0.005, 0.01, 0.02, 0.05] and not 0.0001)
- `sealed-plant-030` [spec -> ok] "sixteen bytes. false alarm rate: point oh five."  gold `{"active_nodes": ["reactor", "separator", "stripper", "compressor_utilities"], "byte_budget": 16, "target_far": 0.05}`  got `{"byte_budget": 16, "target_far": 0.005, "active_nodes": ["reactor", "separator", "stripper", "compressor_utilities"]}`
- `sealed-plant-041` [spec -> refused] "Messages of 8 bytes; use every node apart from the separator; false-alarm budget half of one percent."  gold `{"active_nodes": ["reactor", "stripper", "compressor_utilities"], "byte_budget": 8, "target_far": 0.005}`  got `null` (target_far must be one of [0.001, 0.005, 0.01, 0.02, 0.05] and cannot be calculated as hal)

**keyword/regex parser**: 110 of 174 items not as expected. First 12:

- `sealed-squad-001` [spec -> ok] "pls make green forrage thru 250 turns, high caution"  gold `{"caution": "high", "eaters": ["green"], "food_target": 0, "step_limit": 250}`  got `{"eaters": ["green"], "caution": "normal", "step_limit": 250, "food_target": 0}`
- `sealed-squad-002` [spec -> ok] "Blue only. Reckless. One hundred twenty moves. Fifteen pellets."  gold `{"caution": "low", "eaters": ["blue"], "food_target": 15, "step_limit": 120}`  got `{"eaters": ["blue"], "caution": "low", "step_limit": 300, "food_target": 0}`
- `sealed-squad-003` [spec -> ok] "Everybody go foraging! Safety first, we need twenty apples between us."  gold `{"caution": "high", "eaters": ["red", "green", "blue"], "food_target": 20, "step_limit": 300}`  got `{"eaters": ["red", "green", "blue"], "caution": "normal", "step_limit": 300, "food_target": 0}`
- `sealed-squad-004` [spec -> ok] "I'd like green and red to eat; blue can wander. No more than 2 hundred steps."  gold `{"caution": "normal", "eaters": ["red", "green"], "food_target": 0, "step_limit": 200}`  got `{"eaters": ["red", "green", "blue"], "caution": "normal", "step_limit": 300, "food_target": 0}`
- `sealed-squad-005` [spec -> ok] "forage: blue+green | caution=low | steps=75 | food=9"  gold `{"caution": "low", "eaters": ["green", "blue"], "food_target": 9, "step_limit": 75}`  got `{"eaters": ["green", "blue"], "caution": "normal", "step_limit": 300, "food_target": 0}`
- `sealed-squad-006` [spec -> ok] "Hey, quick one. Green is the only one who needs to eat. Keep it to a hundred steps, and don't take chances."  gold `{"caution": "high", "eaters": ["green"], "food_target": 0, "step_limit": 100}`  got `{"eaters": ["green"], "caution": "normal", "step_limit": 300, "food_target": 0}`
- `sealed-squad-007` [spec -> ok] "Mission for the squad: all three feast until 12 are eaten. Aggressive play is fine."  gold `{"caution": "low", "eaters": ["red", "green", "blue"], "food_target": 12, "step_limit": 300}`  got `{"eaters": ["red", "green", "blue"], "caution": "low", "step_limit": 300, "food_target": 0}`
- `sealed-squad-010` [spec -> ok] "Only the green and the blue should hunt, I want a baker's dozen eaten, 140 ticks."  gold `{"caution": "normal", "eaters": ["green", "blue"], "food_target": 13, "step_limit": 140}`  got `{"eaters": ["green", "blue"], "caution": "normal", "step_limit": 140, "food_target": 0}`
- `sealed-squad-011` [spec -> ok] "Squad, listen up: red takes the lead on foraging, solo, for fifty steps, with the target set to ten."  gold `{"caution": "normal", "eaters": ["red"], "food_target": 10, "step_limit": 50}`  got `{"eaters": ["red"], "caution": "normal", "step_limit": 300, "food_target": 0}`
- `sealed-squad-013` [spec -> ok] "Don't send blue. Red and green eat - thirty food and then we stop."  gold `{"caution": "normal", "eaters": ["red", "green"], "food_target": 30, "step_limit": 300}`  got `{"eaters": ["red", "green", "blue"], "caution": "normal", "step_limit": 300, "food_target": 0}`
- `sealed-squad-015` [spec -> refused] "all three eat, ASAP, no more than 55 ticks, risk it"  gold `{"caution": "low", "eaters": ["red", "green", "blue"], "food_target": 0, "step_limit": 55}`  got `null` (no keywords matched)
- `sealed-squad-016` [spec -> ok] "The red one should forage for one hundred and eighty steps. A cautious approach, please."  gold `{"caution": "high", "eaters": ["red"], "food_target": 0, "step_limit": 180}`  got `{"eaters": ["red"], "caution": "high", "step_limit": 300, "food_target": 0}`

## Limitations

- The sealed set is written by one author who also wrote the training templates, the prompts and the label conventions. The n-gram test keeps frames apart, not ideas: the same person's notion of 'indirect' or 'inverted wording' shapes both sides, and the 10 gold clarify items encode one author's view of 'ambiguous'.
- The prompts were written after the sealed set existed (the brief required it first), without running them on it. I avoided phrasing conventions that only the sealed items exercise, but I cannot prove my prompt choices were independent of having seen them.
- Paraphrases are kept only when the 14B model recovers the gold spec, which also skews the surviving phrasing toward what that model reads easily; the 14B is also one of the translators evaluated. The labels themselves never came from a model.
- Round trip uses summaries written by the 14B from each translator's own trace, so a translator is also judged by how well it reads the 14B's summary style; the fixed summary template makes this easier than free-form monitoring text. Fidelity is a regular-expression check.
- Memory: the distilled model is bf16 on one process; 'beside the specialists' assumes they share this base model (the HF/PEFT design) and does not include the specialists' own adapters, KV caches or the CUDA context (reported separately by nvidia-smi).
- Latency was measured on a machine that may have run other work; Ollama VRAM is the GPU's total used memory over the pre-load baseline.
