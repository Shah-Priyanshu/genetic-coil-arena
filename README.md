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
   - code formation ([docs/results/code_formation.md](docs/results/code_formation.md), pilot with n = 6 runs per arm at 600 generations):
     - With an annealed message-entropy bonus (`--entropy-bonus 1`), a code formed in 6/6 runs, against 4/6 without it. Team food was 25.6 ± 2.1 against 18.8 ± 10.2.
     - A sampled centre with annealed temperature also formed a code in 6/6 runs but learned more slowly.
     - Neither difference is significant at this n;
   - next: specialist snakes that each eat one colour of food and a command centre that sees the whole board, learning bandwidth-limited messages, each with a readable envelope and a learned payload.
3. **Translator:** natural language becomes a typed, readable GoalSpec, which the command centre turns into learned payloads. A reverse translator decodes the payloads for monitoring.
4. **A second, non-game domain,** to prove the interfaces are task-agnostic.
5. **Real small models** behind the same specialist interface.

## Layout

```
src/genetic_coil_arena/
  protocol/     envelopes, signing, codecs (arrays <-> envelopes), GoalSpecs   [domain-agnostic]
  runtime/      specialists, bus, router, command centre, trace, demo          [domain-agnostic]
  evaluation/   CommArena harness: baselines, ablations, metrics, statistics   [domain-agnostic]
  learning/     population MLPs and evolution strategies (numpy)              [domain-agnostic]
  experiments/  environment <-> harness glue (squad listening, learned protocol)
  envs/         batched multi-agent API (base) and the squad environment (squad/)
  snake_core/   single-snake game engine (numba) and the neural agent
  evo/          genetic operators, fitness, parallel evaluation, baselines, evolution loop
  persistence/  experiment storage, checkpoints and resume
  analysis/     Snake benchmark harness, multi-seed runs, squad need-to-talk report
  cli/          the `genetic-coil` command
  ui/           pygame viewer
scripts/        experiment scripts (need_to_talk.py, learned_protocol.py, code_formation.py), GPU runner
docs/results/   recorded results
tests/
```

## License

MIT. See [LICENSE](LICENSE).
