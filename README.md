# Genetic Coil Arena

A research lab for **specialist agents that coordinate through a language they learn themselves**. A command centre directs them, and a translator turns natural-language instructions into the command centre's protocol and back.

**Status:** early research. The repo was rebooted in October 2026. The earlier web platform is preserved under the `legacy-platform` git tag.

## Why

The long-term goal is an in-house, task-agnostic ecosystem of specialized models. Each model is good at one job, and they exchange compact machine messages instead of human language. Building that requires answers to open questions:

- **Protocol:** can specialists learn a compact protocol that beats a hand-designed one?
- **Bandwidth:** how much bandwidth do they actually need?
- **Translation:** can a translator map human intent into the protocol reliably, and decode it back for monitoring?
- **Modularity:** can a new specialist join without retraining everything?

Snake is the first lab for these questions:
- experiments take minutes;
- ground truth is exact, because you know who ate what and when;
- failures are visible.

## What works today

- **Simulator:** a fast single-snake simulator compiled with numba. The board can wrap around (torus) or have walls, and a hunger timer ends games where the snake stops eating.
- **Genetic algorithm:** BLX-α crossover, Gaussian mutation, tournament selection and elitism, with parallel evaluation, persistence and resume.
- **Evaluation harness:**
  - random, straight and greedy baselines;
  - held-out seeds;
  - mean ± 95% confidence intervals;
  - multi-seed evolution runs.
- **Pygame viewer** and analysis plots.
- **Protocol and runtime:** signed envelopes with a readable header and a payload, codecs between batched message arrays and envelopes, schema-validated GoalSpecs, and a command centre that routes goals to specialists and logs a JSONL trace (`genetic-coil protocol-demo`).
- **CommArena evaluation harness:** mandatory baselines, multi-seed runs, message ablations and interventions, communication metrics, IQM with bootstrap CIs and Holm-corrected tests.
- **Squad environment:** a batched, numba-compiled cooperative squad of colour-specialist snakes with scripted specialists and centres. Its need-to-talk baselines are in [docs/results/need_to_talk.md](docs/results/need_to_talk.md) (`genetic-coil squad-benchmark`).
- **Plant-floor sensor squad** (`envs/plant/`, a second, non-game domain): four edge sensor nodes on Tennessee Eastman process data uplink byte-limited messages to a gateway centre that diagnoses the fault. Hand-designed PCA/FFT features reach macro-F1 0.76 at 16 B per message over 18 diagnosable faults ([docs/results/plant_bytes_baselines.md](docs/results/plant_bytes_baselines.md)). The data is fetched by `scripts/fetch_tep.py` into a cache outside the repo.
  - Step 2 ([docs/results/plant_bytes_baselines_rieth.md](docs/results/plant_bytes_baselines_rieth.md)) uses the Rieth et al. 2017 runs, with 10 run-level replicates and a calibrated MLP centre at a 5% false-alarm rate, over 17 faults. Hand designs reach macro-F1 0.82 at 1 B, 0.91 at 4 B and 0.95 at 8 B and above. The oracle scores 0.952: at 16 B and above the hand designs match it, so a learned code can only beat them at 8 B or less.
  - Step 3 ([docs/results/plant_learned_codes.md](docs/results/plant_learned_codes.md), goal G2 passed): learned codes beat the best hand design where bandwidth binds. At 1 B, end-to-end codes gain +0.141 and node-supervised codes (fault id + confidence) +0.147, reaching macro-F1 0.969 against 0.823. The designed semantic code does as well as the end-to-end learned format: the gain comes from learning at the sensor, not from inventing a format.
  - Step 4 ([docs/results/plant_novel_faults.md](docs/results/plant_novel_faults.md), goal G2b, inconclusive and leaning kill): with the sensors frozen, can a learned code handle 5 faults it never trained on, given 3 labelled runs each? No learned code beat the designed fault-id code at 1–4 B. The end-to-end code ties (0.832 against 0.837 at 1 B), and the reconstruction and k-means variants do worse. A self-made format bought neither accuracy nor extensibility here.

Current numbers (100 held-out seeds, food eaten per episode):

| Policy | Food |
|---|---|
| Random | 0.07 ± 0.05 |
| Evolved (256 × 300 generations, 3 seeds) | 0.44 ± 0.13 |
| Greedy heuristic | 39.2 ± 3.6 |

Evolution beats random but is far from greedy. Treat this as an open problem, not a solved baseline. These agents were evolved before commit c63e08b fixed a mirrored (east/west) observation window, so the evolved row needs re-measuring.

## Quick start

```bash
pip install -e ".[dev]"
pytest tests/ --ignore=tests/test_viewer.py   # test_viewer.py opens a pygame window

genetic-coil benchmark -n 100                 # baselines, mean ± 95% CI on held-out seeds
genetic-coil evolve -g 60 -p 128              # one evolution run (resume with -e <experiment id>)
genetic-coil multiseed -k 3 -g 60 -p 128 --no-early-stop -o outputs/multiseed
genetic-coil visualize                        # watch a game
genetic-coil protocol-demo --trace trace.jsonl  # specialists coordinated by the command centre
genetic-coil squad-benchmark -n 100           # squad need-to-talk baselines (Markdown report)
genetic-coil learned-protocol --scale smoke   # R6 learned protocol (ES training + evaluation); --help lists flags
```

On Windows, set `PYTHONUTF8=1`. The CLI prints emoji, which crashes a cp1252 console when output is piped.

## Roadmap

1. **Step 0, an honest measuring stick:** done.
2. **Cooperative squad lab:** in progress.
   - done: the protocol and runtime, the CommArena evaluation harness, and the squad environment with scripted baselines (with a small window, a hand-coded 3-bit centre roughly doubles team food);
   - R6, a learned protocol at 3 bits (first CPU result in [docs/results/learned_protocol.md](docs/results/learned_protocol.md); `genetic-coil learned-protocol`): the centre and the specialists are trained jointly with evolution strategies at window radius 3. **There were only n = 4 training runs per condition, so treat this as a pilot.**
     - Learned listeners reading the fixed hand-coded centre ate 30.7 ± 4.6 food per episode. For comparison, the scripted squad ate 18.7 and the silent learned squad 10.3.
     - The jointly learned code reached 22.5 ± 14.8. In 3 of 4 runs it formed a 1 to 1.6-bit direction code, and those runs fell in the hand-coded range. One run never formed a code and stayed at the silent level.
     - **10-run result ([docs/results/learned_protocol_gpu_r3.md](docs/results/learned_protocol_gpu_r3.md), 2000 generations, 20-core PC):** a code formed in 10/10 runs. The learned centre reached 35.4 ± 1.6 food, the hand-coded centre 34.9 ± 2.4 and silent 15.2 ± 1.1. Learned vs hand-coded is +0.5 [−2.1, +3.2], indistinguishable; learned vs silent is +20.2 [+18.4, +22.0]. The learned code matches the hand-coded protocol while using about 1.2 of the 3 bits: it encodes the direction of the receiver's nearest food and drops the near/far distance. Zeroing or shuffling its messages costs about 32 food.
     - With the entropy bonus on the same 10 runs ([docs/results/learned_protocol_gpu_r3_entropy.md](docs/results/learned_protocol_gpu_r3_entropy.md)), team food was 36.1 ± 2.0, a paired difference of +0.6 [−1.9, +3.1] (p = 0.32). At 2000 generations codes form without the bonus, so it makes no detectable difference here.
     - **At r = 5 ([docs/results/learned_protocol_gpu_r5.md](docs/results/learned_protocol_gpu_r5.md), 10 runs, 32 hidden units):** learned 27.3 ± 1.2, hand-coded 27.3 ± 1.4, silent 14.1 ± 1.9 team food. Learned vs hand-coded is +0.1 [−1.7, +1.8]; the learned code again uses about 1.2 bits, mostly direction, and a code formed in 10/10 runs. Communication still adds about 13 food at r = 5, against about 20 at r = 3. The learned snakes here only match the scripted greedy squad (26.5) and fall below r = 3's learned squad (35.4), so the larger window makes the specialists harder to train rather than the channel less useful.
   - add a specialist ([docs/results/add_specialist.md](docs/results/add_specialist.md), 10 runs): a 4th snake joins a frozen trained squad. From random weights it is not cheap under *either* protocol (0/10 learned and 1/10 hand-coded reached 90% of the jointly retrained newcomer within 124% of joint compute). It still learns and listens to the frozen learned language (5.7 food; ablation −5.3). Initialised as a copy of an existing specialist, it joins with zero training under either protocol (9.9 vs joint 9.85 food). So the cost is learning the skill, not the language: grow the ecosystem from shared initialisations, not from scratch.
   - code formation ([docs/results/code_formation.md](docs/results/code_formation.md), pilot with n = 6 runs per arm at 600 generations):
     - With an annealed message-entropy bonus (`--entropy-bonus 1`), a code formed in 6/6 runs, against 4/6 without it. Team food was 25.6 ± 2.1 against 18.8 ± 10.2.
     - A sampled centre with annealed temperature also formed a code in 6/6 runs but learned more slowly.
     - Neither difference is significant at this n;
   - next: specialist snakes that each eat one colour of food and a command centre that sees the whole board, learning bandwidth-limited messages, each with a readable envelope and a learned payload.
3. **Translator:** natural language becomes a typed, readable GoalSpec, which the command centre turns into learned payloads. A reverse translator decodes the payloads for monitoring.
   - R7a vertical slice (`translator/`, `scripts/r7a_vertical_slice.py`): instruction, then translator, validated GoalSpec, command centre, specialists, trace, reverse summary and re-translation, on the squad and the plant floor with one translator. Full run ([docs/results/r7a_vertical_slice.md](docs/results/r7a_vertical_slice.md)): with qwen2.5-coder:14b, exact match on held-out phrasings was 84.7%, just under the G3 kill line of 85%. Most misses were refusals of valid terse or indirect requests. Round trip 100%, summary fidelity 100%, warm p95 1.19 s.
   - R7a step 2 (`translator/policy.py`, `scripts/r7a_step2.py`): a refusal policy (parse first, validate after; documented defaults; a new `clarify` outcome for ambiguous requests) and a Qwen3-1.7B LoRA distilled from templates plus verified qwen3-coder:30b paraphrases, to be judged on a sealed 174-item test set (`experiments/data/r7a_sealed_test.jsonl`, SHA-256 pinned). Runs are queued as `r7a2-*` jobs; the report lands in `docs/results/r7a_step2.md`. **Result** ([docs/results/r7a_step2.md](docs/results/r7a_step2.md), G3 passed): on a sealed set of 174 hand-written items, qwen2.5-coder:14b reaches 95.9% exact match. gemma4:e4b reaches 92.4% at 4.3 GB, and the distilled Qwen3-1.7B LoRA 82.8%.
   - R7b (goal G7, `experiments/r7b_goal_conditioned.py`, `scripts/r7b_goal_conditioned.py`): can a learned centre follow a `squad.mission` GoalSpec (which colours eat, caution), with the goal travelling only through the 3-bit messages, including on eater/caution combinations never trained on? Five conditions (blind, goal-conditioned centre, goal-conditioned centre and specialists, hand-coded spec-aware centre, R7a scripted wrapper), 8 ES runs each. Queued as `r7b-*` jobs; report `docs/results/r7b_goal_conditioned.md`. **Result** ([docs/results/r7b_goal_conditioned.md](docs/results/r7b_goal_conditioned.md), G7 partial): on held-out goals, learned goal-aware centres beat the goal-blind one. Scores: 7.1 with the goal only in messages, 15.6 with the goal also at the specialists, against −3.5 goal-blind. The hand-coded goal-aware centre still leads at 20.6. That first run capped learned centres at about 1000–1500 generations; the matched 5000-generation rerun ([docs/results/r7b_goal_conditioned_5k.md](docs/results/r7b_goal_conditioned_5k.md), **G7 passed**) gives 18.5 with the goal only in messages and 19.8 with it at the specialists too, against −3.4 goal-blind and 21.7 hand-coded (85% and 91%). At 10k generations ([docs/results/r7b_goal_conditioned_10k.md](docs/results/r7b_goal_conditioned_10k.md)) the goal-in-messages centre reaches 91% of the hand-coded one, no longer significantly below it.
4. **A second, non-game domain,** to prove the interfaces are task-agnostic: in progress (step 1: hand-designed byte baselines done).
5. **Real small models** behind the same specialist interface.
   - R9 (two LoRA specialists on a shared Qwen3-1.7B, latent messages vs JSON): kill criterion not met, so JSON stays the default.
     - Hidden-state messages got 0% exact match.
     - A KV-prefix handoff matched JSON and saved about 43% latency, but is 11 MiB per record against 104 B, so it suits only same-host specialists.
     - A single model matched the pipeline, so the task doesn't force a split.
     - One task, one base, 3 seeds: [docs/results/r9_latent_vs_json.md](docs/results/r9_latent_vs_json.md).
   - R10 step 1 (goal G4, `envs/logtriage/`, `scripts/r10_log_triage.py`): a task where no model can read the whole input. RCAEval RE2 incidents (Sock Shop, Online Boutique; per-service logs and metrics, labelled root-cause service and fault type) are 400-700 times a 4k-token context. Compared at a matched centre context: a single model with naive truncation, a single model with a strong heuristic selection, a long-context single model, and a team of per-service specialists with typed JSON or free-text messages, plus non-LLM baselines. Zero-shot Qwen3-1.7B (and Ollama qwen2.5-coder:14b); training is step 2. Code, data pipeline and baselines done; the evaluation is queued as `r10-*` runner jobs, results in `docs/results/r10_log_triage.md`. **Result** ([docs/results/r10_log_triage.md](docs/results/r10_log_triage.md), G4 kill): LLM teams (4–23% top-1) lose to one model given strong selection (64–69%). Both lose to non-LLM BARO on service metrics (88%): the typed messages did not carry the per-service anomaly scores that decide the root cause.
   - R10 step 2 (goal G4b, `envs/logtriage/team.py`, `experiments/log_triage2*.py`, `scripts/r10_step2.py`): a heterogeneous team with typed messages that carry the decisive signal. Per-service metric specialists (code; schema-validated anomaly score, onset, top metrics, tens of bytes) and per-service LLM log specialists (fault hypothesis, confidence, evidence templates, reading only their own logs around the reported onset; qwen2.5-coder:14b and Qwen3-1.7B zero-shot), a rule centre or a 14B LLM centre. Judged on joint accuracy (root cause and fault type) against BARO and the step-1 single models; the hybrid's fault-type gain is the real test, since a metric-only team with a rule centre is distributed BARO. The hybrid also runs through `CommandCentre`, envelopes and `TraceLog`. **Result** ([docs/results/r10_step2.md](docs/results/r10_step2.md), G4b kill): no hybrid team beats BARO on joint accuracy (45.0). The 14B log specialists add nothing, Qwen3-1.7B's cost 1.7 points, and log reports cut the 14B centre's joint accuracy by 13 points. The typed metric reports alone let a 14B centre reach 46.7 joint, against 29.2 when it reads raw data. Fault type (disk, loss, socket) is the bottleneck.
   - R10 step 3 (goal G4c, `envs/logtriage/trained.py`, `experiments/log_triage3*.py`, `scripts/r10_step3.py`): trained fault-type specialists in the typed team. A numpy classifier on per-kind metric features and a Qwen3-1.7B LoRA on the service's own logs each send six fault probabilities in a schema-validated report; a rule centre ranks by the metric score and takes the fault from the reports. Training uses Train Ticket plus SS/OB repetition 1, with selection by leave-one-system-out cross-validation, and the test split is never fitted on. **Result** ([docs/results/r10_step3.md](docs/results/r10_step3.md), G4c pass): the metric specialist's team reaches 68.3 joint against BARO's 45.0 (+23.3 [+15.0, +31.7]); trained on Train Ticket alone, +18.3. Socket, loss and disk recover. The LoRA log specialist stays at chance, so on RE2 the signal is in the metrics.
   - R10 step 4 (goal G4d, `envs/logtriage/logsignal.py`, `team4.py`, `experiments/log_triage4*.py`, `scripts/r10_step4.py`): LLM log specialists on RCAEval RE3, whose code-level faults show up in the logs (new warning/error templates in the root service in 60–100% of incidents). **Result** ([docs/results/r10_step4.md](docs/results/r10_step4.md), G4d kill): every LLM team is significantly below the non-LLM fusion of metric scores and new-template counts (80.6 top-1): 14B 64.5 (LLM centre) and 62.9 (rule centre), Qwen3-1.7B 50.0, gemma4:e4b 59.7. A plain new-template count alone reaches 77.4. The 14B flags the faulty service but also 20–42% of the healthy ones, so its reports don't rank.
   - R10 step 5 (goal G4e, `envs/logtriage/rerank.py`, `experiments/log_triage5*.py`, `scripts/r10_step5.py`): an LLM as final reader of the non-LLM fusion's top-3 shortlist on RE3 (14B, gemma4:e4b, 30B). **Result** ([docs/results/r10_step5.md](docs/results/r10_step5.md), G4e kill): every reader scores below the fusion's 80.6 top-1 (14B 77.4, gemma4:e4b 74.2, 30B 72.6). The readers keep the fusion's top-1 85-89% of the time, and their overrides break more correct answers than they fix, mostly on Train Ticket.
   - R11 ecosystem v0 (goal G8, `experiments/r11_*.py`, `scripts/r11_ecosystem.py`, `envs/logtriage/catalogue.py`): the whole chain on real data. One shared translator picks the domain (squad, plant, or the new incident domain) and extracts a GoalSpec; `rca.investigate` names an RE2 test incident by ticket (`INC-1001` to `INC-1120`) or by its real alert time; the command centre runs the G4c trained team on it; the reverse translator summarises the trace and the summary is translated again. Judged on a sealed set of 130 hand-written requests (`experiments/data/r11_sealed_test.jsonl`, SHA-256 pinned). **Result** ([docs/results/r11_ecosystem.md](docs/results/r11_ecosystem.md), G8 kill on one criterion): the 14B translator gets 95.5% exact match, 100% routing, 95.3% round trip and 100% summary fidelity at 4.4 s p95. But four requests that give a day without a month were read as February while their gold says January, so the task score from natural language falls 3.6 points below gold, which is significant. A v2 (goal G8b) fixes date resolution (the desk resolves partial dates, not the translator), makes the summary state the requested top-k and uses a fresh sealed set of 142 requests (`experiments/data/r11b_sealed_test.jsonl`); **Result** ([docs/results/r11b_ecosystem.md](docs/results/r11b_ecosystem.md), G8b passed): the 14B gets 92.5% exact match, 100% routing, 98.3% round trip and 100% summary fidelity at 4.8 s p95, and the task score from natural language equals the gold run. gemma4:e4b (83.3%) and the keyword parser (82.5%) fail.
   - R12 growth (goals G9 and G10, jobs `r12-*`, `experiments/r12_*.py`, `scripts/r12_growth.py`): a log-novelty specialist joins the incident team without retraining the rest (G9), and the desk extends to Train Ticket and the 62 RE3 incidents without prompt changes (G10, sealed set `experiments/data/r12_sealed_test.jsonl`, 107 requests, SHA-256 pinned). **Result** ([docs/results/r12_growth.md](docs/results/r12_growth.md)): G9 kill. The newcomer adds 22.6 points top-1 on RE3 but costs 3.3 (significant) on RE2, all on packet-loss faults, where the callers log the errors. G10 is not passed and not killed: Train Ticket joins with no prompt edits (100% routing, 90.6% exact on its specs, NL task score equal to gold on the new incidents), but the 14B's habit of adding unstated years keeps overall exact match at 88%.

## Layout

```
src/genetic_coil_arena/
  protocol/     envelopes, signing, codecs (arrays <-> envelopes), GoalSpecs   [domain-agnostic]
  runtime/      specialists, bus, router, command centre, trace, demo          [domain-agnostic]
  evaluation/   CommArena harness: baselines, ablations, metrics, statistics   [domain-agnostic]
  learning/     population MLPs and evolution strategies (numpy)              [domain-agnostic]
  experiments/  environment <-> harness glue (squad listening, learned protocol)
  envs/         batched multi-agent API (base), the squad environment (squad/), the plant-floor sensor squad (plant/)
  snake_core/   single-snake game engine (numba) and the neural agent
  evo/          genetic operators, fitness, parallel evaluation, baselines, evolution loop
  persistence/  experiment storage, checkpoints and resume
  analysis/     Snake benchmark harness, multi-seed runs, squad need-to-talk report
  cli/          the `genetic-coil` command
  ui/           pygame viewer
scripts/        experiment scripts (need_to_talk.py, learned_protocol.py, code_formation.py, plant_baselines.py, fetch_tep.py), GPU runner
docs/results/   recorded results
tests/
```

## License

MIT. See [LICENSE](LICENSE).
