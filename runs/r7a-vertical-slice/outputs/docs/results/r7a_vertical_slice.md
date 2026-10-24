# R7a vertical slice: natural language to GoalSpec to command centre to trace to summary

Generated 2026-10-24 16:28 on PC (Python 3.13.7). Command: `python scripts/r7a_vertical_slice.py --out docs/results/r7a_vertical_slice`.

## What was run

One translator class (`translator.Translator`) serves two domains. A domain supplies only data: a `SpecDefinition` in a shared `SpecRegistry`, a system prompt with few-shot examples, an executor that honours the GoalSpec by wrapping the public squad or plant APIs, and the facts a summary must state.

- **Squad** (`squad.mission`): which colours forage, caution (low/normal/high), step limit 50-300, food target 0-40. Non-eater snakes wander; caution sets the wander rate; the target ends the mission. Task score: eater food (capped at the target) plus 0.5 per eater alive at the end, mean over the seeds.
- **Plant floor** (`plant.diagnose`): byte budget (8/16/51), target false-alarm rate (0.1% to 5%), active nodes. The feature encoder and the calibrated centre are trained for the ordered budget and rate with the silenced nodes' windows zeroed. Task score: macro-F1 over the fault classes, mean over fit seeds.

Evaluation set: 160 instructions (16 invalid or unsatisfiable, 10%). Gold GoalSpecs come from the generator's own parameters (phrase banks and templates), never from a model. `indist` uses plain templated phrasings close to the few-shot examples; `heldout` uses terse, verbose, indirect and number-in-words phrasings, `everyone except X` negations and other unit spellings that appear in no few-shot.

Seeds: squad missions run on 8 held-out episodes (seeds from 2,000,000,000), plant runs on 5 fit-and-replay repeats. Each model call is retried once with the validation error before a typed failure.

## Verdict

**KILL criterion hit** (primary model qwen2.5-coder:14b).

| Criterion | Target | Measured | Result |
|---|---|---|---|
| G3 exact match, held-out valid items | >= 95% (kill < 85%) | 84.7% (n=72, 95% CI 74.7%-91.2%) | KILL |
| G3 task score within CI of gold-spec run | paired diff >= 0 or inside its CI, both domains | squad: NL 6.60 vs gold 6.85, diff -0.25 ± 0.09; plant: NL 0.51 vs gold 0.56, diff -0.05 ± 0.00 | not met |
| G3 round trip (reverse summary re-translated) | >= 90% (kill < 70%) | 100.0% (n=133, 95% CI 97.2%-100.0%) | PASS |
| G6 summary field fidelity | >= 90% (kill < 75%) | 100.0% of facts (all facts in 100.0% of summaries) | PASS |
| G3 warm translator p95 | < 2 s | 1.19 s | PASS |
| G3 fits in 12 GB | <= 12 GB | 8.6 GB peak | PASS |

## GoalSpec accuracy

Exact match is over the instructions with a gold spec (params equal after sorting unordered arrays); 95% Wilson intervals.

| System | In-distribution | Held-out | Held-out squad | Held-out plant |
|---|---|---|---|---|
| qwen2.5-coder:14b | 97.2% (90%-99%, n=72) | 84.7% (75%-91%, n=72) | 88.9% (75%-96%, n=36) | 80.6% (65%-90%, n=36) |
| gemma4:e4b | 81.9% (72%-89%, n=72) | 58.3% (47%-69%, n=72) | 69.4% (53%-82%, n=36) | 47.2% (32%-63%, n=36) |
| keyword/regex parser | 100.0% (95%-100%, n=72) | 16.7% (10%-27%, n=72) | 13.9% (6%-29%, n=36) | 19.4% (10%-35%, n=36) |

Per-field accuracy, held-out split:

| System | active_nodes | byte_budget | caution | eaters | food_target | step_limit | target_far |
|---|---|---|---|---|---|---|---|
| qwen2.5-coder:14b | 86.1% | 86.1% | 88.9% | 88.9% | 88.9% | 88.9% | 80.6% |
| gemma4:e4b | 50.0% | 50.0% | 69.4% | 69.4% | 69.4% | 69.4% | 47.2% |
| keyword/regex parser | 75.0% | 61.1% | 58.3% | 94.4% | 50.0% | 55.6% | 36.1% |

Per-field accuracy, in-distribution split:

| System | active_nodes | byte_budget | caution | eaters | food_target | step_limit | target_far |
|---|---|---|---|---|---|---|---|
| qwen2.5-coder:14b | 94.4% | 94.4% | 100.0% | 100.0% | 100.0% | 100.0% | 94.4% |
| gemma4:e4b | 75.0% | 75.0% | 88.9% | 88.9% | 88.9% | 88.9% | 75.0% |
| keyword/regex parser | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

## Refusals

A refusal is any outcome without a validated GoalSpec (the model refused, or its reply never validated). Gold refusals are the invalid requests.

| System | Precision | Recall | TP / FP / FN | Statuses |
|---|---|---|---|---|
| qwen2.5-coder:14b | 59.3% | 100.0% | 16 / 11 / 0 | {'ok': 133, 'refused': 27} |
| gemma4:e4b | 27.6% | 100.0% | 16 / 42 / 0 | {'ok': 102, 'refused': 58} |
| keyword/regex parser | 66.7% | 37.5% | 6 / 3 / 10 | {'invalid': 5, 'ok': 151, 'refused': 4} |

## Task score driven from natural language vs the gold spec

Both runs go through the command centre on the same episodes. An instruction whose translation failed scores 0 on every seed. Mean over instructions per seed, then mean ± 95% t-interval over seeds; the last column is the paired difference (NL minus gold) over seeds.

| System | Split / domain | n | NL | Gold | NL - gold | Within CI |
|---|---|---|---|---|---|---|
| qwen2.5-coder:14b | indist/squad | 36 | 6.83 ± 0.86 | 6.83 ± 0.86 | 0.00 ± 0.00 | yes |
| qwen2.5-coder:14b | indist/plant | 36 | 0.55 ± 0.01 | 0.59 ± 0.01 | -0.04 ± 0.00 | no |
| qwen2.5-coder:14b | heldout/squad | 36 | 6.38 ± 0.76 | 6.88 ± 0.85 | -0.49 ± 0.17 | no |
| qwen2.5-coder:14b | heldout/plant | 36 | 0.48 ± 0.01 | 0.54 ± 0.01 | -0.06 ± 0.00 | no |
| qwen2.5-coder:14b | all/squad | 72 | 6.60 ± 0.69 | 6.85 ± 0.75 | -0.25 ± 0.09 | no |
| qwen2.5-coder:14b | all/plant | 72 | 0.51 ± 0.01 | 0.56 ± 0.01 | -0.05 ± 0.00 | no |
| gemma4:e4b | indist/squad | 36 | 6.28 ± 0.82 | 6.83 ± 0.86 | -0.54 ± 0.16 | no |
| gemma4:e4b | indist/plant | 36 | 0.42 ± 0.01 | 0.59 ± 0.01 | -0.17 ± 0.00 | no |
| gemma4:e4b | heldout/squad | 36 | 4.92 ± 0.66 | 6.88 ± 0.85 | -1.96 ± 0.32 | no |
| gemma4:e4b | heldout/plant | 36 | 0.27 ± 0.01 | 0.54 ± 0.01 | -0.27 ± 0.01 | no |
| gemma4:e4b | all/squad | 72 | 5.60 ± 0.59 | 6.85 ± 0.75 | -1.25 ± 0.22 | no |
| gemma4:e4b | all/plant | 72 | 0.34 ± 0.01 | 0.56 ± 0.01 | -0.22 ± 0.00 | no |
| keyword/regex parser | indist/squad | 36 | 6.83 ± 0.86 | 6.83 ± 0.86 | 0.00 ± 0.00 | yes |
| keyword/regex parser | indist/plant | 36 | 0.59 ± 0.01 | 0.59 ± 0.01 | 0.00 ± 0.00 | yes |
| keyword/regex parser | heldout/squad | 36 | 8.24 ± 1.37 | 6.88 ± 0.85 | 1.36 ± 0.89 | yes |
| keyword/regex parser | heldout/plant | 36 | 0.57 ± 0.01 | 0.54 ± 0.01 | 0.03 ± 0.00 | yes |
| keyword/regex parser | all/squad | 72 | 7.53 ± 0.97 | 6.85 ± 0.75 | 0.68 ± 0.44 | yes |
| keyword/regex parser | all/plant | 72 | 0.58 ± 0.01 | 0.56 ± 0.01 | 0.02 ± 0.00 | yes |

## Reverse translator (primary model only)

The summary is written from the executed trace (GoalSpec, envelope headers, outcome), then translated back into a GoalSpec by the forward translator. Round trip: the reconstruction equals the forward spec. Field fidelity: the share of required facts (every parameter plus the outcome score) the summary states without a contradiction, checked by regular expressions against the GoalSpec and the outcome.

| Group | n | Round trip (95% CI) | Field fidelity | All facts in summary | Per fact |
|---|---|---|---|---|---|
| all | 133 | 100.0% (97%-100%) | 100.0% | 100.0% | active_nodes 100%, byte_budget 100%, caution 100%, eaters 100%, food_target 100%, score 100%, step_limit 100%, target_far 100% |
| indist | 70 | 100.0% (95%-100%) | 100.0% | 100.0% | active_nodes 100%, byte_budget 100%, caution 100%, eaters 100%, food_target 100%, score 100%, step_limit 100%, target_far 100% |
| heldout | 63 | 100.0% (94%-100%) | 100.0% | 100.0% | active_nodes 100%, byte_budget 100%, caution 100%, eaters 100%, food_target 100%, score 100%, step_limit 100%, target_far 100% |
| squad | 68 | 100.0% (95%-100%) | 100.0% | 100.0% | caution 100%, eaters 100%, food_target 100%, score 100%, step_limit 100% |
| plant | 65 | 100.0% (94%-100%) | 100.0% | 100.0% | active_nodes 100%, byte_budget 100%, score 100%, target_far 100% |

Summary latency p50 / p95: 0.93 / 1.06 s.

Sample summaries:

- `squad-indist-00` (round trip ok, fidelity 100%): The blue snake was ordered to forage at caution level normal, with a step limit of 200 and a food target of 8; the squad finished with mission score 5.9.
- `squad-indist-01` (round trip ok, fidelity 100%): The red snake was ordered to forage at caution level normal, with a step limit of 50 and a food target of 8; the squad finished with mission score 2.9.
- `squad-indist-02` (round trip ok, fidelity 100%): The red snake was ordered to forage at caution level high, with a step limit of 60 and a food target of 15; the squad finished with mission score 2.3.
- `squad-indist-03` (round trip ok, fidelity 100%): The red snake was ordered to forage at caution level low, with a step limit of 150 and a food target of 30; the squad finished with mission score 6.6.
- `squad-indist-04` (round trip ok, fidelity 100%): The green snake was ordered to forage at caution level normal, with a step limit of 250 and a food target of 15; the squad finished with mission score 9.5.
- `squad-indist-05` (round trip ok, fidelity 100%): The red and blue snakes were ordered to forage at caution level normal, with a step limit of 100 and a food target of 10; the squad finished with mission score 6.2.

## Latency and memory

Cold is the first call after unloading every model (it includes the model load); warm is every later call. Item latency includes a retry when one happened.

| System | Cold first call | Warm item p50 / p95 | Warm call p50 / p95 | Peak VRAM over baseline | Max prompt tokens |
|---|---|---|---|---|---|
| qwen2.5-coder:14b | 4.8 s | 0.89 / 1.19 s | 0.89 / 1.19 s | 8.6 GB (peak 9.4 GB total, baseline 0.8 GB) | 684 |
| gemma4:e4b | 11.4 s | 0.43 / 0.54 s | 0.43 / 0.54 s | 4.3 GB (peak 5.0 GB total, baseline 0.8 GB) | 704 |
| keyword/regex parser | - | - | - | - | - |

## Failures of the primary model

13 of 160 instructions were wrong (a valid request not matched exactly, or an invalid one accepted). First 13:

- `squad-heldout-12` [refused] "red: you have seventy-five moves, treat the walls with respect and avoid losses, the squad needs 24 bites between them."  gold `{"eaters": ["red"], "caution": "high", "step_limit": 75, "food_target": 24}`  got `null` (this is not an order for the squad)
- `squad-heldout-17` [refused] "green: within one hundred fifty ticks."  gold `{"eaters": ["green"], "caution": "normal", "step_limit": 150, "food_target": 0}`  got `null` (this is not an order for the squad)
- `squad-heldout-19` [refused] "Here is what I am hoping for today: the job of red and blue is to hunt for food. Also, a dozen in total; avoid any danger; within sixty ticks."  gold `{"eaters": ["red", "blue"], "caution": "high", "step_limit": 60, "food_target": 12}`  got `null` (ticks are not a valid unit of time for this mission; please use steps instead)
- `squad-heldout-24` [refused] "red: stop after 120 ticks, two dozen items."  gold `{"eaters": ["red"], "caution": "normal", "step_limit": 120, "food_target": 24}`  got `null` (this is not an order for the squad)
- `plant-indist-04` [refused] "A budget of 8 bytes, false alarm rate 0.5%."  gold `{"byte_budget": 8, "target_far": 0.005, "active_nodes": ["reactor", "separator", "stripper", "compressor_utilities"]}`  got `null` (0.5% is not an allowed false alarm rate (0.001, 0.005, 0.01, 0.02, 0.05))
- `plant-indist-12` [refused] "A budget of 16 bytes, false alarm rate 0.1%."  gold `{"byte_budget": 16, "target_far": 0.001, "active_nodes": ["reactor", "separator", "stripper", "compressor_utilities"]}`  got `null` (0.1% is not an allowed false alarm rate (0.001, 0.005, 0.01, 0.02, 0.05))
- `plant-heldout-01` [ok] "Good morning. For the fault diagnosis run, I would like you to configure things as follows: a 16 B payload; and false alarms should stay near one in two hundred. Please confirm when done."  gold `{"byte_budget": 16, "target_far": 0.005, "active_nodes": ["reactor", "separator", "stripper", "compressor_utilities"]}`  got `{"byte_budget": 16, "target_far": 0.01, "active_nodes": ["reactor", "separator", "stripper", "compressor_utilities"]}`
- `plant-heldout-09` [ok] "Thinking about the plant diagnosis, I suppose the alarm budget for normal operation is five in a hundred, and also payloads of sixteen bytes at most, and also everything from the reactor and nobody else, if that is possible."  gold `{"byte_budget": 16, "target_far": 0.05, "active_nodes": ["reactor"]}`  got `{"byte_budget": 16, "target_far": 0.01, "active_nodes": ["reactor"]}`
- `plant-heldout-13` [refused] "all nodes bar the reactor / a 16 B payload."  gold `{"byte_budget": 16, "target_far": 0.01, "active_nodes": ["separator", "stripper", "compressor_utilities"]}`  got `null` (this is not a request about the plant system)
- `plant-heldout-14` [refused] "Thinking about the plant diagnosis, I suppose each node may send eight bytes, and also tolerate roughly a tenth of a percent of normal readings raising an alarm, and also rely on nothing but the compressor and utilities node, if that is possible."  gold `{"byte_budget": 8, "target_far": 0.001, "active_nodes": ["compressor_utilities"]}`  got `null` (compressor and utilities is one node, not two)
- `plant-heldout-15` [refused] "each node may send fifty-one bytes / tolerate roughly two in a hundred of normal readings raising an alarm / rely on nothing but the reactor."  gold `{"byte_budget": 51, "target_far": 0.02, "active_nodes": ["reactor"]}`  got `null` (this is not a request about the plant system)
- `plant-heldout-20` [refused] "rely on nothing but the separator / the alarm budget for normal operation is five percent."  gold `{"byte_budget": 16, "target_far": 0.05, "active_nodes": ["separator"]}`  got `null` (five percent is not an allowed false alarm rate (0.001, 0.005, 0.01, 0.02, 0.05))
- `plant-heldout-21` [refused] "the alarm budget for normal operation is a tenth of a percent / everything from the separator and nobody else / each node may send fifty-one bytes."  gold `{"byte_budget": 51, "target_far": 0.001, "active_nodes": ["separator"]}`  got `null` (0.1% is not an allowed false alarm rate (0.001, 0.005, 0.01, 0.02, 0.05))

## Limitations

- The evaluation set is synthetic (templates plus hand-written phrase banks); its held-out split changes style, not the label conventions, which the system prompts state. A different author could phrase things differently, and the prompts' conventions (for example `low` caution for 'take risks') are mine.
- The prompts were adjusted once after a 20-instruction smoke run drawn from this same set (the rate-conversion rule of the plant prompt, and the status-report examples that let the forward translator read a summary). No other tuning was done on the set, but those 20 items are not fully fresh.
- The keyword baseline was written against the in-distribution phrasing, so its in-distribution score is an upper bound for rules of that kind; its held-out score is what such rules do on new styles.
- The few-shot examples of the summary pack fix the summary's phrasing (it names the parameters in a fixed pattern), which both helps the fact check and makes the round trip easier than free-form monitoring text would be. Fidelity is checked by regular expressions, not by a human or a second model.
- Squad non-eaters wander (blind to their own food) rather than hold still, because snakes cannot stop. Plant silencing trains the centre on zeroed windows for the silenced nodes; the trace of a plant task holds only the goal and result envelopes, because the sensor uplinks run on the batched array path.
- Latency was measured on a machine that may run other jobs; the VRAM peak is the GPU's total used memory over the pre-load baseline and includes other processes that started in between.
