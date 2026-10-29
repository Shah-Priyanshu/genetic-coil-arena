# R7c: is a goal carried only by learned messages reliable to train? (goal G7c)

Generated 2026-10-29 by job `r7c-zz-report2` from `r7c-train-c` (runs 16–25) and `r7c-train-d` (runs 26–35), pinned to 24dc6320.

**G7c verdict: PASS.** With the restart rule (discard a run whose best training-goal validation mission is below 12), the kept (b) runs reach 95% of the hand-coded centre (d) on held-out goals (pass ≥ 90%, kill < 85%).

- Kept − (d): -1.24 ± 1.69 (Welch 95% CI), Mann-Whitney p = 0.0302; n = 16 kept against 16.
- Kept − (a): +25.18 ± 1.03.
- Flagged 0 of 20 fresh runs; pooled with G7's runs 0-15, 2 of 36 (Wilson 95% CI 2-18%).

<!-- analysis:start -->
**What G7c shows: a goal carried only by the 3-bit messages trains reliably.** None of the 20 fresh runs failed. The lowest held-out score was 17.7, and the lowest best training-goal validation was 17.8, well above the 12.0 flag.

- **The restart rule never fired.** The kept sample is simply runs 16–31, so the pass does not depend on the rule. Its value on new failures is untested. Looking back, it would have removed exactly G7's two failures (validation 5.7 and 8.2, held-out 6.6 and 8.0) and nothing else.
- **The failure rate is low:** 2 of 36 (b) runs over G7 and G7c, 5.6% (Wilson 2–18%). G7's 2 of 16 looks like an unlucky draw, not a property of the condition.
- **(b) sits slightly below the hand-coded centre (d), by about 5%.**
  - Kept runs: −1.24 ± 1.69 (Welch CI includes zero, Mann-Whitney p = 0.03).
  - All 20 fresh runs: −1.17 ± 1.67, p = 0.04.
  - Seen goals: 18.09 against (d)'s 18.97.
- **(b) now matches G7's (c), the goal at the centre and the specialists** (21.31): kept (b) − (c) = +0.52 ± 1.42, p = 0.40. G7's gap between (c) and (b) came entirely from the two failed runs. Once (b) trains, putting the goal only in the messages costs nothing.
- **Pooled over all 36 (b) runs:** held-out 20.92 (91% of (d)). The 34 runs above the flag average 21.72 (94%).
- **The seed sets are comparable.** Fresh runs are scored on their own episode seeds. Baselines there match G7's: goal-blind greedy 3.01 against 3.50, scripted wrapper 0.81 against 0.87.

**When a failure becomes visible (post hoc, from the training curves).** The table shows the running best training-goal validation at each checkpoint.

| Generation | 3,000 | 5,000 | 10,000 |
|---|---|---|---|
| G7's two failures | 4.0, 6.8 | 4.8, 7.2 | 5.7, 8.2 |
| Lowest of the other 34 runs | 2.2 (G7 run 1), 7.2 (run 18) | 13.4 (run 31) | 17.8 |

- **At 3,000 generations it is too early.** Two runs that ended at 18.6 and 22.2 were still below the failures.
- **At 5,000 the cut at 12 separates all 36 runs.** A check there would halve the cost of a failed run, about 7 of its 14 hours.

This was chosen after seeing the data, so it is a suggestion for future training, not a tested rule.

**Compute:** each run took about 14 hours on one worker, about 280 CPU-hours for the 20. The first pair of jobs lost about 4 hours of 20 workers to the runner's 240-minute default timeout. This pair ran about 7 hours alongside 20 orphaned workers from those jobs, which `proc.kill()` had left alive. The runner's kill-tree fix addresses that.
<!-- analysis:end -->

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
