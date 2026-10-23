# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A research lab for specialist agents that coordinate through a learned communication protocol, directed by a command centre and fronted by a natural-language translator. The roadmap is in README.md:

1. step 0 (done);
2. cooperative snake squad;
3. translator;
4. a second, non-game domain;
5. real small models.

The repo was rebooted in 2026-10: the old FastAPI platform, the PyTorch toy engine and `archive/` were deleted. They are recoverable from the `legacy-platform` git tag. Don't resurrect web code without being asked.

The Snake simulator is a **lab**, not the product. New capabilities should be reusable across domains (see "Conventions for new code").

## Setup and commands

```bash
pip install -e ".[dev]"
pytest tests/ --ignore=tests/test_viewer.py      # test_viewer.py opens a pygame window
pytest tests/test_snake_game.py::TestSnakeGame::test_toroidal_wrapping
genetic-coil benchmark -n 100 [--config experiments/<id>/config.yaml --agent <file>]   # baselines + saved agents, mean ± 95% CI on held-out seeds
genetic-coil evolve -g 60 -p 128                 # resume with -e <experiment id>
genetic-coil multiseed -k 3 -g 60 -p 128 --no-early-stop -o outputs/multiseed   # curves_ci.csv + final_benchmark.csv
genetic-coil protocol-demo [--trace trace.jsonl]  # arithmetic specialists via the command centre
genetic-coil squad-benchmark -n 100 [-r 5] [-o docs/results/need_to_talk.md]   # squad need-to-talk baselines
ruff check <files>; black <files>
```

- **Windows: set `PYTHONUTF8=1`.** The CLI prints emoji. When output is piped (including from Claude's shell), a cp1252 console crashes with `UnicodeEncodeError`.
- **numba cache:**
  - `@njit(cache=True)` writes caches into `src/**/__pycache__`.
  - Old `py38` caches there may still reference enum classes and fail with `No module named 'src'`. Fix it by deleting `*.nbi` and `*.nbc`.
  - New code must pass plain ints into numba helpers, never enums.
  - Set `NUMBA_CACHE_DIR` to a short path if a long path breaks caching (Windows' 260-character limit).
- **Windows process spawning:** fitness evaluation uses `ProcessPoolExecutor`, so any script that calls `evolve()` needs an `if __name__ == "__main__":` guard.
- **pytest settings:** `filterwarnings = error` turns warnings into failures, except UserWarning and DeprecationWarning. `--strict-markers` is on; the markers are `slow`, `integration` and `benchmark`.
- **Lint is not clean.** ruff reports many pre-existing issues, so run ruff and black on the files you touch, not the whole tree. CI (`.github/workflows/ci.yml`) runs ruff, mypy and pytest on a 3-OS × Python 3.9–3.12 matrix and does not pass yet.

## Architecture

Data flow:
1. `Config` is passed to `EvolutionArena.evolve()`.
2. `BatchEvaluator` hands genomes to `FitnessEvaluator`, which runs them in a persistent process pool.
3. `run_episode` plays `SnakeGame` with an agent.
4. `compute_fitness` scores each episode, producing `GenomeStats`.
5. `create_next_generation` builds the next population.

Components:
- **Game (`snake_core/game.py`).**
  - The board wraps around by default; `walls=True` makes the edges deadly instead.
  - `starvation_steps` (default 100) kills a snake that goes too long without food.
  - Rewards come from `GameConfig`. Build games with `SnakeGame.from_config()`. Each game has its own random generator.
  - Actions are relative to the snake's heading: `STRAIGHT`, `LEFT`, `RIGHT`.
  - The observation is an egocentric `(2r+1)²` window, plus the normalized length, plus food dx/dy; `get_observation_size()` gives its size. With walls, off-board cells read as body.
  - Keep the hot paths numba-compatible: numpy arrays and scalars only.
- **Agent (`snake_core/agent.py`).**
  - A fixed MLP with two hidden layers and 3 outputs.
  - The genome is a flat float32 vector in the order `W1,b1,W2,b2,W3,b3`.
  - Genome size is computed in two places, `SnakeAgent.genome_size` and `EvolutionArena._calculate_genome_size`. Change both together.
- **GA operators (`evo/genome.py`).** BLX-α crossover, per-gene Gaussian mutation, tournament selection and elitism. DEAP is used only for its `creator` classes.
- **Fitness (`evo/fitness.py`).**
  - `evaluation.fitness_function` selects the mode:
    - `food` (default): food eaten, with survival only breaking ties;
    - `legacy`: `score + 0.1*steps`;
    - `return`: the sum of game rewards.
  - Baselines and evolved agents share `run_episode` and `compute_fitness`.
  - Worker failures are logged and retried in-process, and a batch where every episode fails raises.
  - The old "survival trap" (random snakes scoring about 50 under legacy fitness) only exists with `starvation_steps: 0`, `max_steps: 500` and `fitness_function: legacy`.
- **Baselines (`evo/baselines.py`).** Random, straight, and greedy (heads for the food, but never into a move that dies next step). Held-out benchmark seeds start at 2,000,000,000.
- **Harness (`analysis/benchmark.py`).**
  - `benchmark_policies`; re-exports `mean_ci`, whose one implementation is `evaluation/stats.py`.
  - `load_agent_file`, which accepts any saved agent format.
  - `select_best_genome`, which re-ranks the hall of fame on validation seeds.
  - `run_multi_seed_evolution`.
- **Outputs (`evo/arena.py`).** Results go to `config.logging.output_dir` (default `outputs/`): `generation_stats.csv` (including `mean_steps` and `starvation_rate`), `summary.json` (including `fitness_function` and `random_baseline_fitness`), `hall_of_fame.pkl`, checkpoints and `config.yaml`. The random baseline is measured at the start of each run.
- **Persistence (`persistence/`).** `ModelPersistence` keeps experiment directories under `experiments/`. `PersistentEvolutionArena` adds checkpointing and resume on top of `EvolutionArena`.
- **Squad report (`analysis/need_to_talk.py`).** Need-to-talk baselines behind `squad-benchmark` and `scripts/need_to_talk.py`. Results go to `docs/results/` (tracked).
- **`evo/advanced.py` is unused.** It covers novelty search, speciation and adaptive parameters. Its `SpeciationEvolution` reads a nonexistent `genome.parameters`. For quality-diversity work, prefer the pyribs library.

New packages (the squad-lab stack):
- **Protocol (`protocol/`).** `Envelope` (readable header + payload, HMAC `signing`), `codecs` between message arrays and envelopes, `GoalSpec` + `SpecRegistry` (JSON Schema via `jsonschema`, `MinimalValidator` fallback), `tensor_store`.
- **Runtime (`runtime/`).** `Specialist`/`ClockAdapter`, `InProcessBus`, routers, `SpecialistRegistry`, `CommandCentre`, JSONL `TraceLog` (training triples). `runtime/demo.py` is the arithmetic demo (`protocol-demo`).
- **Evaluation (`evaluation/`).** CommArena harness: `runner` (tasks, mandatory baselines, multi-seed runs, message ablations), `interventions`, communication `metrics`, `stats` (t-CI, IQM bootstrap, Mann-Whitney + Holm, A12), `report`.
- **Environments (`envs/`).** `envs.base` is the batched multi-agent API (`MultiAgentEnv`, `EnvSpec`, `MessageSpec`). `envs/squad/` is the cooperative squad: numba engine, `SquadEnv`, scripted specialists and centres, `run_episodes`.
- **Lazy top level.** `genetic_coil_arena/__init__.py` resolves `Config`, `EvolutionArena`, `SnakeGame` and `SnakeAgent` on first access, so importing `protocol`, `runtime` or `evaluation` never loads `snake_core` or numba (a subprocess test checks it). Keep it that way.

## Conventions for new code

- **Keep the protocol and agent layers domain-agnostic.** Packages for the protocol, specialists, the command centre and the translator must not import `snake_core` or any environment package. Domain-specific code lives in environment packages that implement the shared interfaces.
- **Two layers:**
  - a batched numpy/numba fast path for training: messages and observations are arrays;
  - an object layer (envelopes, specialists, command centre) for orchestration and, later, real models.
  
  Keep converters between the two explicit.
- **`protocol/`, `runtime/` and `evaluation/` are domain-agnostic.** AST tests enforce their import rules; don't add `snake_core`, `evo`, `envs` or `analysis` imports there.
- **Message arrays:** discrete codes are int64 `(B, R, L)`, continuous vectors float32 `(B, R, D)`. Convert to envelopes only via `protocol.codecs`.
- **Goals:** every goal must pass `SpecRegistry.make` or `SpecRegistry.validate`. Never hand-build a `GoalSpec` that skips validation.
- **Baselines:** environments register their baselines with `evaluation.register_baseline`; `run_experiment` refuses a task with no registered baseline unless `strict=False`.
- **Report results with evidence.** Every claim needs the baselines, multiple seeds and confidence intervals from the step-0 harness. Before trusting any numbers, check that evaluation isn't failing silently.

## Known issues

- **Mirrored Snake observation (fixed in c63e08b).** Until that commit the egocentric window had east/west mirrored, so every agent evolved earlier saw a mirrored world on half its headings. Re-evolve before trusting older numbers or saved agents.
- **Evolution is weak.** Under the default (food) fitness, evolved agents reached only about 0.4 food per episode, against greedy's about 39 (measured 2026-10, before the mirroring fix, which may explain part of the gap). Other likely causes are a sparse signal, noisy fitness from only 3 episodes, and different seeds for each genome. Not yet re-measured.
- **Winner's curse.** `PersistentEvolutionArena` saves `hall_of_fame[0]` (ranked on noisy training scores) as `best_agent.pkl`. `multiseed` re-ranks on validation seeds instead.
- **Stray folder:** `PersistentEvolutionArena` also creates an unused `outputs/` folder in the current directory.
- **Stub:** `genetic-coil replay`.
