# Goals

Measurable targets on the way to the end goal: an ecosystem of specialist models coordinated by a command centre and fronted by a translator. Every target is judged with the CommArena harness (baselines, multiple seeds or replicates, 95% CIs, message ablations). Each goal states in advance what would count as failure (kill).

## Where the evidence points (2026-10-24)

- **Learned codes only match good hand-designed codes.**
  - R6, squad, at r = 3 and r = 5: learned 35.4 against hand-coded 34.9 team food, then 27.3 against 27.3.
  - R8 step 2: hand designs reach the oracle at 8 B and above.
- **Raw latent messages between LLM specialists fail.** In R9, hidden states gave 0% exact match. A KV-cache handoff is lossless and cuts latency by 43%, but is about 100,000× larger than JSON.
- **Growth works by copying.** In R7, a newcomer initialised from an existing specialist joins at zero training cost under either protocol. A from-scratch newcomer is costly under both, because it must learn the skill, not the language.

Working design:
- a **typed, machine-native core protocol**: schema-validated GoalSpecs and messages;
- **KV handoff** as a fast path between specialists on the same host;
- **learned compression** only where bandwidth binds;
- **new specialists initialised from existing ones**: LoRA from LoRA on the shared base.

## Goal ladder

| Id | Goal | Target (pass) | Kill | Track / lane | Status |
|---|---|---|---|---|---|
| G0 | Keep the PC busy and every push tested | CPU lane busy ≥ 70% and GPU lane ≥ 50% of hours over a week; a local CI run within 2 h of every push to the jobs branch; no lane idle > 2 h while the backlog is non-empty | — | infra | in progress |
| G1 | A specialist can join without retraining the rest | A newcomer reaches 90% of the jointly trained newcomer's food at ≤ 30% of joint compute, with old specialists within CI of before | > 30% under both protocols | R7, squad (cloud) | met for copy-initialised newcomers, failed from scratch (`docs/results/add_specialist.md`) |
| G2 | Learned codes win where bandwidth binds | At 1, 2 or 4 B per uplink, a learned code beats the best hand design's macro-F1 at matched 5% false alarms, with a 95% CI over 10 replicates excluding 0 (headroom: +0.13 at 1 B, +0.05 at 4 B) | No budget shows a significant win | R8 step 3, plant / CPU | queued as `r8s3-*` jobs |
| G3 | End-to-end vertical slice | Natural language → translator → GoalSpec → command centre → specialists → trace → reverse translator. Targets: GoalSpec exact match ≥ 95% on held-out paraphrases; task score from natural language within CI of the hand-written-spec run; round trip (reverse summary re-translated) gives the same GoalSpec ≥ 90%; warm translator p95 < 2 s; fits in 12 GB | Exact match < 85%, or round trip < 70% | R7a, translator / GPU | next, after R9 merges |
| G4 | Specialisation that is forced, with real models | A log-triage task whose total input exceeds one model's context budget: specialists + centre beat the best single model at the same budget by ≥ 5 points (CI excludes 0). Then learned compressed messages against typed JSON at matched accuracy: ≥ 30% fewer tokens | Single model within CI of the team | R10, real models / GPU | planned |
| G5 | Same-host fast path | KV handoff as an optional runtime transport on G4's task: ≥ 30% end-to-end latency saving with no accuracy loss (CI) | Accuracy loss or < 15% saving | R9 follow-up / GPU | planned |
| G6 | Monitoring | The reverse translator turns traces into readable summaries; ≥ 90% field fidelity on a sample checked against the GoalSpec and ground truth | < 75% | R7a / R10 | planned |

## Compute plan

The runner (`scripts/gpu_runner.py`, `docs/gpu_runner.md`) executes queued jobs on the PC (28 threads, 12 GB GPU, 32 GB RAM).

| Lane | What runs there | Budget |
|---|---|---|
| cpu | numpy and numba sweeps: R8 learned codes, squad ES runs, CI test runs | ≤ 24 workers in total |
| gpu | LLM work: translator evaluations (Ollama), LoRA training and evaluation (Qwen3-1.7B) | one job; `set_per_process_memory_fraction(0.85)` |

Both lanes run at once. Sweeps are split into many jobs, so that the backlog keeps the PC busy while no Claude session is active.
