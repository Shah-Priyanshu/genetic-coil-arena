# Squad: can a new specialist join a learned language? (R7, PILOT)

> **Pilot (smoke preset).** Small budgets on 28 CPUs to check the pipeline and calibrate the GPU job; the joint reference is under-trained here, so the compute ratios are not the result.

- Date: 2026-10-28
- Commit: ac91fbc8 (code; this report is committed on top of it)
- Compute: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, 28 logical cores, 1 worker processes. Training wall time 0.0 min (0.00 CPU-hours), evaluation 0.0 min.
- Source squads: R6 r = 3 runs 0–1 (`C:\Users\priya\AppData\Local\Temp\pytest-of-unknown\pytest-183\test_driver_end_to_end_and_res0\params.npz`, sha256 f1511983b767…): 3 snakes, one shared specialist network per squad, trained 2000 ES generations × population 128 × 8 episodes.
- Training: ES population 4 × 1 episodes per member; generations: newcomer from scratch 2, newcomer copied 2, joint retraining 2. Validation every 2 generations on 2 episodes.
- Held-out evaluation: 3 test episodes per run (R6's test seeds, now with 4 snakes). Runs are the independent units for every CI/test.

Reproduce with `python scripts/add_specialist.py --scale smoke --runs 2 --generations 2 --copy-generations 2 --joint-generations 2 --population 4 --episodes-per-member 1 --fitness newcomer --workers 1 --eval-episodes 3 --out C:/Users/priya/AppData/Local/Temp/pytest-of-unknown/pytest-183/test_driver_end_to_end_and_res0/report/add.md`.

## Verdict

- **Kill criterion:** KILL: most from-scratch newcomers under the frozen learned language did not reach the threshold within a budget of 100% of joint-retraining compute (median over runs, > 30%, so the true ratio is larger): the ecosystem needs a fixed typed core protocol.

<!-- discussion -->

## Design

- **Newcomer:** a 4th snake with a 4th colour joins a trained 3-snake squad (21×21, 2 food per colour, so the board gains 2 food items and a body). Only its network trains (`EvolutionStrategy` mask = 1 on its block); the old specialists' shared network and the centre stay frozen. Newcomer fitness = its own return (its food + 0.01 per step alive): with the team return the frozen snakes' ~30 food per episode varies between ES members once the newcomer perturbs their trajectories and swamps its first food items. Harm to the old snakes is measured below. The joint squad trains on the team return, as in R6.
- **The frozen centre addresses the newcomer unchanged:** both R6 networks are receiver-agnostic (the centre sees each receiver's own-colour food egocentrically; specialists see own/other planes, not colours), so with N = 4 the centre encodes colour 3's food for snake 3 in the same code (tested: permuting snakes permutes the codes).
- **Conditions:** (1) frozen learned centre + learned specialists; (2) frozen `HandCodedCentre` + its learned listeners; (3) joint retraining of all 4 snakes (one shared network, as in R6) + a learned centre from scratch; (4) silent squad. Newcomer init: `scratch` (random) or `copy` (the squad's own shared weights, a zero-shot clone because the network is colour-agnostic).
- **Compute metric:** reference = (3)'s per-snake food on the test episodes, per run; threshold = 90% of it. Compute to threshold = training episodes until the trailing 3-checkpoint mean of the validation curve reaches it (newcomer's food; per-snake mean for (3)). Ratio = newcomer / joint, run k vs run k; a newcomer that never reaches it counts as ∞. Kill if the median ratio of (1)-scratch > 30%.
- **Listening:** the harness's ablations applied to the newcomer's channel only (other receivers' messages untouched).

## Compute to reach the jointly trained newcomer's food

| condition | init | runs | reached threshold | median compute vs joint (to threshold) | runs within kill ratio | median gens to threshold | median gens to 90% of old snakes' food |
|---|---|---|---|---|---|---|---|
| learned | scratch | 2 | 0/2 | > 100% (budget spent) | 0% | never | never |
| hand-coded | scratch | 2 | 0/2 | > 100% (budget spent) | 0% | never | never |
| joint | - | 2 | 2/2 | 100% | n/a | 1 | n/a |
| silent | scratch | 2 | 0/2 | > 100% (budget spent) | 0% | never | never |

Per run:

| condition | init | run | ref food (joint, per snake) | threshold | gens to threshold | episodes | compute vs joint | vs joint full budget | wall-time vs joint |
|---|---|---|---|---|---|---|---|---|---|
| hand-coded | scratch | 0 | 0.08 | 0.07 | never | - | n/a | never (∞) | never (∞) |
| hand-coded | scratch | 1 | 0.08 | 0.07 | never | - | > 100% (budget spent) | never (∞) | never (∞) |
| joint | - | 0 | 0.08 | 0.07 | 0 | 0 | n/a | n/a | n/a |
| joint | - | 1 | 0.08 | 0.07 | 2 | 8 | n/a | n/a | n/a |
| learned | scratch | 0 | 0.08 | 0.07 | never | - | n/a | never (∞) | never (∞) |
| learned | scratch | 1 | 0.08 | 0.07 | never | - | > 100% (budget spent) | never (∞) | never (∞) |
| silent | scratch | 0 | 0.08 | 0.07 | never | - | n/a | never (∞) | never (∞) |
| silent | scratch | 1 | 0.08 | 0.07 | never | - | > 100% (budget spent) | never (∞) | never (∞) |

## Newcomer food on held-out episodes

Mean ± 95% t-CI over runs. Score = the newcomer's (snake 3's) food per episode. Baselines put a scripted newcomer into run s's frozen learned squad.

| system | role | newcomer food | old snakes' food (sum of 3) | team food |
|---|---|---|---|---|
| (1) learned+newcomer[scratch] | system | 0.00 ± 0.00 | 0.17 ± 2.12 | 0.17 ± 2.12 |
| (2) hand-coded+newcomer[scratch] | system | 0.00 ± 0.00 | 0.33 ± 4.24 | 0.33 ± 4.24 |
| (3) joint-retrain | system | 0.00 ± 0.00 | 0.33 ± 0.00 | 0.33 ± 0.00 |
| (4) silent+newcomer[scratch] | system | 0.00 ± 0.00 | 0.33 ± 4.24 | 0.33 ± 4.24 |
| newcomer:random | baseline | 0.00 ± 0.00 | 0.17 ± 2.12 | 0.17 ± 2.12 |
| newcomer:no-comm-greedy | baseline | 2.83 ± 19.06 | 0.17 ± 2.12 | 3.00 ± 16.94 |
| newcomer:oracle | baseline | 20.00 ± 29.65 | 0.17 ± 2.12 | 20.17 ± 31.77 |

Comparisons vs baselines (Mann-Whitney over runs, Holm):

```
Task: squad-add-specialist-r3  (seeds: 2)
Baselines: newcomer:random, newcomer:no-comm-greedy, newcomer:oracle

condition                         role      n  mean ± 95% CI    IQM [95% CI]           
--------------------------------  --------  -  ---------------  -----------------------
(1) learned+newcomer[scratch]     system    2  0.000 ± 0.000    0.000 [0.000, 0.000]   
(2) hand-coded+newcomer[scratch]  system    2  0.000 ± 0.000    0.000 [0.000, 0.000]   
(3) joint-retrain                 system    2  0.000 ± 0.000    0.000 [0.000, 0.000]   
(4) silent+newcomer[scratch]      system    2  0.000 ± 0.000    0.000 [0.000, 0.000]   
newcomer:random                   baseline  2  0.000 ± 0.000    0.000 [0.000, 0.000]   
newcomer:no-comm-greedy           baseline  2  2.833 ± 19.059   2.833 [1.333, 4.333]   
newcomer:oracle                   baseline  2  20.000 ± 29.648  20.000 [17.667, 22.333]

system                            intervention    ablated mean  drop [95% CI]         p (Holm)  A12    listening
--------------------------------  --------------  ------------  --------------------  --------  -----  ---------
(1) learned+newcomer[scratch]     zero            0.000         0.000 [0.000, 0.000]  1.000     0.500  no       
(1) learned+newcomer[scratch]     shuffle_batch   0.000         0.000 [0.000, 0.000]  1.000     0.500  no       
(1) learned+newcomer[scratch]     random_matched  0.000         0.000 [0.000, 0.000]  1.000     0.500  no       
(2) hand-coded+newcomer[scratch]  zero            0.000         0.000 [0.000, 0.000]  1.000     0.500  no       
(2) hand-coded+newcomer[scratch]  shuffle_batch   0.000         0.000 [0.000, 0.000]  1.000     0.500  no       
(2) hand-coded+newcomer[scratch]  random_matched  0.000         0.000 [0.000, 0.000]  1.000     0.500  no       
(3) joint-retrain                 zero            0.000         0.000 [0.000, 0.000]  1.000     0.500  no       
(3) joint-retrain                 shuffle_batch   0.000         0.000 [0.000, 0.000]  1.000     0.500  no       
(3) joint-retrain                 random_matched  0.000         0.000 [0.000, 0.000]  1.000     0.500  no       
(4) silent+newcomer[scratch]      zero            0.000         0.000 [0.000, 0.000]  1.000     0.500  no       
(4) silent+newcomer[scratch]      shuffle_batch   0.000         0.000 [0.000, 0.000]  1.000     0.500  no       
(4) silent+newcomer[scratch]      random_matched  0.000         0.000 [0.000, 0.000]  1.000     0.500  no       

comparison                                                   mean diff  p      p (Holm)  A12  
-----------------------------------------------------------  ---------  -----  --------  -----
(1) learned+newcomer[scratch] vs newcomer:random             0.000      1.000  1.000     0.500
(1) learned+newcomer[scratch] vs newcomer:no-comm-greedy     -2.833     0.221  1.000     0.000
(1) learned+newcomer[scratch] vs newcomer:oracle             -20.000    0.221  1.000     0.000
(2) hand-coded+newcomer[scratch] vs newcomer:random          0.000      1.000  1.000     0.500
(2) hand-coded+newcomer[scratch] vs newcomer:no-comm-greedy  -2.833     0.221  1.000     0.000
(2) hand-coded+newcomer[scratch] vs newcomer:oracle          -20.000    0.221  1.000     0.000
(3) joint-retrain vs newcomer:random                         0.000      1.000  1.000     0.500
(3) joint-retrain vs newcomer:no-comm-greedy                 -2.833     0.221  1.000     0.000
(3) joint-retrain vs newcomer:oracle                         -20.000    0.221  1.000     0.000
(4) silent+newcomer[scratch] vs newcomer:random              0.000      1.000  1.000     0.500
(4) silent+newcomer[scratch] vs newcomer:no-comm-greedy      -2.833     0.221  1.000     0.000
(4) silent+newcomer[scratch] vs newcomer:oracle              -20.000    0.221  1.000     0.000
```

## Does the newcomer listen? (ablations of its channel)

Drop = intact − ablated newcomer food, paired by run; p = Wilcoxon, Holm per system.

| system | intervention (newcomer's channel only) | intact | ablated | drop [95% CI] | p (Holm) | listening |
|---|---|---|---|---|---|---|
| (1) learned+newcomer[scratch] | zero | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 1.000 | no |
| (1) learned+newcomer[scratch] | shuffle_batch | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 1.000 | no |
| (1) learned+newcomer[scratch] | random_matched | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 1.000 | no |
| (2) hand-coded+newcomer[scratch] | zero | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 1.000 | no |
| (2) hand-coded+newcomer[scratch] | shuffle_batch | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 1.000 | no |
| (2) hand-coded+newcomer[scratch] | random_matched | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 1.000 | no |
| (3) joint-retrain | zero | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 1.000 | no |
| (3) joint-retrain | shuffle_batch | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 1.000 | no |
| (3) joint-retrain | random_matched | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 1.000 | no |
| (4) silent+newcomer[scratch] | zero | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 1.000 | no |
| (4) silent+newcomer[scratch] | shuffle_batch | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 1.000 | no |
| (4) silent+newcomer[scratch] | random_matched | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 1.000 | no |

## Are the old specialists' food preserved?

The three old snakes' summed food per held-out episode, mean ± 95% CI over runs; differences paired by run (t-CI, Wilcoxon p).

| source squad | newcomer init | 3-snake squad | 4 snakes, random newcomer | 4 snakes, trained newcomer | trained − 3-snake [95% CI] (p) | trained − random newcomer [95% CI] (p) |
|---|---|---|---|---|---|---|
| learned | scratch | 0.00 ± 0.00 | 0.17 ± 2.12 | 0.17 ± 2.12 | 0.17 ± 2.12 (1.000) | 0.00 ± 0.00 (1.000) |
| hand-coded | scratch | 0.17 ± 2.12 | 0.33 ± 4.24 | 0.33 ± 4.24 | 0.17 ± 6.35 (1.000) | 0.00 ± 0.00 (1.000) |
| silent | scratch | 0.17 ± 2.12 | 0.67 ± 0.00 | 0.33 ± 4.24 | 0.17 ± 6.35 (1.000) | -0.33 ± 4.24 (1.000) |

## Learning curves

Validation food (mean over runs) at quantiles of each run's curve (median generation in brackets). Full curves in `add_curves.csv`.

| condition | init | metric | q0 | q10 | q25 | q50 | q75 | q100 |
|---|---|---|---|---|---|---|---|---|
| learned | scratch | val_newcomer_food | 0.00 (g0) | 0.00 (g0) | 0.00 (g0) | 0.00 (g0) | 0.00 (g2) | 0.00 (g2) |
| hand-coded | scratch | val_newcomer_food | 0.00 (g0) | 0.00 (g0) | 0.00 (g0) | 0.00 (g0) | 0.00 (g2) | 0.00 (g2) |
| joint | - | val_per_snake_food | 0.06 (g0) | 0.06 (g0) | 0.06 (g0) | 0.06 (g0) | 0.19 (g2) | 0.19 (g2) |
| silent | scratch | val_newcomer_food | 0.00 (g0) | 0.00 (g0) | 0.00 (g0) | 0.00 (g0) | 0.00 (g2) | 0.00 (g2) |
