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

Current numbers (100 held-out seeds, food eaten per episode):

| Policy | Food |
|---|---|
| Random | 0.07 ± 0.05 |
| Evolved (256 × 300 generations, 3 seeds) | 0.44 ± 0.13 |
| Greedy heuristic | 39.2 ± 3.6 |

Evolution beats random but is far from greedy. Treat this as an open problem, not a solved baseline.

## Quick start

```bash
pip install -e ".[dev]"
pytest tests/ --ignore=tests/test_viewer.py   # test_viewer.py opens a pygame window

genetic-coil benchmark -n 100                 # baselines, mean ± 95% CI on held-out seeds
genetic-coil evolve -g 60 -p 128              # one evolution run (resume with -e <experiment id>)
genetic-coil multiseed -k 3 -g 60 -p 128 --no-early-stop -o outputs/multiseed
genetic-coil visualize                        # watch a game
```

On Windows, set `PYTHONUTF8=1`. The CLI prints emoji, which crashes a cp1252 console when output is piped.

## Roadmap

1. **Step 0, an honest measuring stick:** done.
2. **Cooperative squad lab:**
   - specialist snakes that each eat one colour of food;
   - a command centre that sees the whole board;
   - bandwidth-limited messages, each with a readable envelope and a learned payload.
3. **Translator:** natural language becomes a typed, readable GoalSpec, which the command centre turns into learned payloads. A reverse translator decodes the payloads for monitoring.
4. **A second, non-game domain,** to prove the interfaces are task-agnostic.
5. **Real small models** behind the same specialist interface.

## Layout

```
src/genetic_coil_arena/
  snake_core/   game engine (numba) and the neural agent
  evo/          genetic operators, fitness, parallel evaluation, baselines, evolution loop
  persistence/  experiment storage, checkpoints and resume
  analysis/     benchmark harness (confidence intervals, multi-seed runs)
  cli/          the `genetic-coil` command
  ui/           pygame viewer
tests/
```

## License

MIT. See [LICENSE](LICENSE).
