# Squad: the trained protocol through the object layer (task B)

- Date: 2026-10-23
- Commit: 44d2792 (code; this report is committed on top of it)
- Machine: x86_64, 4 logical cores, one process, BLAS threads = 1
- Policies: learned-centre+learned-spec (R6, r = 3), the best-validation checkpoints of training runs 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 from `runs/r6-c-r3` (`learned_protocol_params.npz` on the `gpu-results` branch; the report is `docs/results/learned_protocol_gpu_r3.md`).
- Episodes: 100 held-out test episodes per run (run s plays seeds 3,500,000,000 + s × 10,000 + i, the R6 test seeds), T = 300.

Reproduce: `git show origin/gpu-results:runs/r6-c-r3/artifacts/learned_protocol_params.npz > params_r3.npz`, then `python scripts/squad_runtime.py --params params_r3.npz --runs 0-9 --episodes 100`.

## Design

- **One episode = one task.** A `squad.forage` v1 GoalSpec (`episode_seed`, optional `max_steps` ≤ 1000, optional `assignment` = specialist name per colour; result schema `team_food`, `food`, `steps`, `survivors`, `return`) is made with `SpecRegistry.make` and submitted to the runtime `CommandCentre`, which routes it to `squad-lead` and validates the result against the schema.
- **Parties on the `InProcessBus`:** `command-centre` (runtime), `squad-lead` (the learned centre network, wrapped as a `ClockAware` specialist) and `squad-snake-0..2` (one specialist per snake, all with the trained specialist network). Each runs in a `ClockAdapter`; the centre and the lead sign every envelope (HMAC-SHA256) and every receiver verifies.
- **Messages:** every tick the lead computes one code per live snake from the full view and sends it as a `codes` envelope (V = 8, L = 1) built only by `protocol.codecs.codes_to_envelopes`; a snake decodes it with `envelopes_to_codes` after its adapter verified the signature. Bus delivery takes one round, which is the fast path's 'read one step later'.
- **World:** `SquadWorld` wraps `SquadEnv` with B = 1 as the snakes' sensors and actuators and steps once every live snake acted. It never carries messages (`env.step` gets no messages; the snakes read their code only from envelopes).
- **Matched randomness:** the fast path's `LearnedSpecialist` draws the Gumbel action noise of all three snakes from one generator per episode. The world draws the same noise each step and hands snake k its row (common random numbers), so the object layer must reproduce the B = 1 fast path exactly; the batched fast path (B = all episodes, as the R6 evaluation runs it) draws different noise and can only agree in distribution.
- **Trace:** `runtime.TraceLog` (JSONL) holds every envelope (goal, codes, result), the task state events, a compact situation snapshot per step (heads, headings, food, health, alive, score; its `clock` equals that step's code envelopes) and a `signature_rejected` event per dropped envelope.

## Equivalence (acceptance test)

- **Step-for-step identity:** 1000 / 1000 episodes identical to the B = 1 fast path: 0 code mismatches, 0 heard-code mismatches and 0 action mismatches over 292,197 steps (live snakes), and the same team food in every episode.
- **Envelopes:** 680,882 envelopes crossed the bus and all verified (0 signature rejections outside the tamper test).

Team food per episode, mean ± 95% t-CI over training runs (each run's score is its mean over the test episodes):

| path | team food |
|---|---|
| object layer (runtime, envelopes) | 35.27 ± 1.80 |
| fast path, B = 1, same seeds and noise | 35.27 ± 1.80 |
| fast path, batched (R6 evaluation style) | 35.62 ± 1.79 |

Object layer − batched fast path, paired by run: -0.35 ± 0.94 (the CI contains 0 when the two agree in distribution).

| run | identical | object | fast B = 1 | batched | object ep/s | B = 1 ep/s | batched ep/s |
|---|---|---|---|---|---|---|---|
| 0 | 100/100 | 34.41 | 34.41 | 36.44 | 3.60 | 14.7 | 455.5 |
| 1 | 100/100 | 36.40 | 36.40 | 37.94 | 3.98 | 15.1 | 413.0 |
| 2 | 100/100 | 33.18 | 33.18 | 35.51 | 3.79 | 15.4 | 340.4 |
| 3 | 100/100 | 35.52 | 35.52 | 34.76 | 3.78 | 14.2 | 483.2 |
| 4 | 100/100 | 34.76 | 34.76 | 34.25 | 3.66 | 13.9 | 486.0 |
| 5 | 100/100 | 39.09 | 39.09 | 38.81 | 3.87 | 15.4 | 461.5 |
| 6 | 100/100 | 29.84 | 29.84 | 30.02 | 3.97 | 14.6 | 609.7 |
| 7 | 100/100 | 36.80 | 36.80 | 35.09 | 3.75 | 16.0 | 472.0 |
| 8 | 100/100 | 37.14 | 37.14 | 38.07 | 3.53 | 14.4 | 441.1 |
| 9 | 100/100 | 35.56 | 35.56 | 35.36 | 3.67 | 14.6 | 582.1 |

## Overhead

- Object layer: 3.76 episodes/s (0.91 ms per episode step: 3 signed envelopes, verification, decoding, one trace snapshot, and per-snake network calls).
- Fast path, B = 1: 14.8 episodes/s (3.9× the object layer).
- Fast path, batched: 463.1 episodes/s (123× the object layer).

The object layer is for orchestration, logging and (later) real models, not training; it is fast enough to record traces and run demos.

## Tamper test

A compromised sidecar on the lead (`TamperingAdapter`) rewrites the signed envelope to `squad-snake-0` at steps [50, 150] of each of 20 episodes of run 0: the code becomes (sent + 1) mod 8 after signing (only while that snake is alive).

- Tampered envelopes: 31; rejected by the snake's `ClockAdapter`: 31 (reason: payload digest mismatch); `signature_rejected` trace events: 31. Every task still completed (['completed']).
- **Fallback (`hold-last-verified-code`):** a dropped envelope counts as not sent, so the snake keeps the last verified code of the current task (0 before any), the environment's own convention for the `message` observation. 20 / 20 tampered episodes are identical, step for step, to the fast path with a hook that re-delivers the previous code at the tampered steps.
- Effect: with the fallback, 4 / 20 episodes acted differently from the intact episode (a stale code instead of the fresh one); had the forged code been accepted, 5 / 20 would have.

## Sample trace

`docs/results/squad_runtime_trace_sample.jsonl` (457 KiB, 813 records): two episodes of run 0's learned protocol capped at `max_steps` = 100 through the GoalSpec; the second uses a permuted `assignment` and has one tampered envelope (1 `signature_rejected` event). `TraceLog.triples()` gives 2 (situation, spec, messages) triples with spec ['squad.forage'], [101, 101] situation snapshots and [302, 302] envelopes. `TraceLog.triples()` keeps only the last snapshot per trace; per-step snapshots are in `situations(trace_id)` and pair with envelopes by `clock`.
