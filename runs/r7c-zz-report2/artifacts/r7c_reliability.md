# R7c: is a goal carried only by learned messages reliable to train? (goal G7c)

**G7c verdict: PASS.** With the restart rule (discard a run whose best training-goal validation mission is below 12), the kept (b) runs reach 95% of the hand-coded centre (d) on held-out goals (pass ≥ 90%, kill < 85%).

- Kept − (d): -1.24 ± 1.69 (Welch 95% CI), Mann-Whitney p = 0.0302; n = 16 kept against 16.
- Kept − (a): +25.18 ± 1.03.
- Flagged 0 of 20 fresh runs; pooled with G7's runs 0-15, 2 of 36 (Wilson 95% CI 2-18%).

## Held-out goals

| Runs | Mean ± 95% CI | % of (d) | Min |
|---|---|---|---|
| (b) + rule, kept runs (n = 16) | 21.83 ± 1.03 | 95% | 17.7 |
| (b), first 16 fresh runs, no rule | 21.83 ± 1.03 | 95% | 17.7 |
| (b), all 20 fresh runs | 21.90 ± 0.98 | 95% | 17.7 |
| (b), G7 runs 0-15 | 19.68 ± 2.68 | 85% | 6.6 |
| (d), G7 runs 0-15 | 23.07 ± 1.42 | 100% | 17.1 |
| (a), G7 runs 0-15 | -3.35 ± 0.14 | -15% | -3.6 |

(d) and (a) are G7's runs 0-15 (`docs/results/r7b_goal_conditioned_16runs.json`); fresh runs are evaluated on their own episode seeds, so comparisons are Welch over independent runs.

Seen goals (context): kept (b) 18.09 ± 0.88, (d) 18.97 ± 1.28.

## Fresh runs

| Run | Best training-goal validation | Held-out | Seen | Kept |
|---|---|---|---|---|
| 16 | 21.7 | 23.9 | 19.9 | yes |
| 17 | 17.8 | 17.7 | 14.8 | yes |
| 18 | 22.2 | 24.5 | 20.7 | yes |
| 19 | 20.2 | 22.6 | 18.8 | yes |
| 20 | 22.5 | 24.1 | 19.9 | yes |
| 21 | 21.4 | 23.7 | 18.9 | yes |
| 22 | 18.3 | 19.5 | 16.2 | yes |
| 23 | 21.5 | 21.7 | 20.0 | yes |
| 24 | 20.3 | 22.7 | 17.7 | yes |
| 25 | 20.5 | 20.3 | 17.2 | yes |
| 26 | 20.7 | 21.4 | 17.1 | yes |
| 27 | 20.4 | 22.3 | 18.0 | yes |
| 28 | 21.2 | 21.3 | 17.4 | yes |
| 29 | 19.3 | 21.4 | 18.1 | yes |
| 30 | 20.2 | 22.8 | 19.1 | yes |
| 31 | 19.2 | 19.1 | 15.8 | yes |
| 32 | 18.9 | 18.9 | 15.7 | spare |
| 33 | 22.3 | 23.9 | 21.5 | spare |
| 34 | 22.5 | 25.3 | 20.5 | spare |
| 35 | 19.1 | 20.6 | 16.7 | spare |

Baselines on the fresh runs' episode seeds (held-out mean): random-actions -3.73, greedy+hand-coded (goal-blind) 3.01, (e) scripted-wrapper (R7a) 0.81.

Training settings: `{"radius": 3, "generations": 10000, "population": 64, "reps": 1, "sigma": 0.05, "lr": 0.03, "weight_decay": 0.005, "spec_hidden": 24, "centre_hidden": 24, "eval_every": 50, "val_reps": 2, "max_steps": 300}`.

Reproduce: `python scripts/r7c_reliability.py --from-json runs:r7c-train-*` (after the `r7c-train-*` jobs).
