# Squad: can a new specialist join a learned language? (R7, PILOT)

> **Pilot (cpu preset).** Small budgets on 4 CPUs to check the pipeline and calibrate the GPU job; the joint reference is under-trained here, so the compute ratios are not the result.

- Date: 2026-10-23
- Commit: 1a7bfad (code; this report is committed on top of it)
- Compute: Intel(R) Xeon(R) Processor @ 2.10GHz, 4 logical cores, 3 worker processes. Training wall time 48.2 min (2.41 CPU-hours), evaluation 1.5 min.
- Source squads: R6 r = 3 runs 0–1 (`r6-c-r3 artifacts/learned_protocol_params.npz from the gpu-results branch, saved locally`, sha256 fb60873dacba…): 3 snakes, one shared specialist network per squad, trained 2000 ES generations × population 128 × 8 episodes.
- Training: ES population 64 × 8 episodes per member; generations: newcomer from scratch 250, newcomer copied 0, joint retraining 1000. Validation every 10 generations on 64 episodes.
- Held-out evaluation: 100 test episodes per run (R6's test seeds, now with 4 snakes). Runs are the independent units for every CI/test.

Reproduce with `python scripts/add_specialist.py --scale cpu --runs 2 --generations 250 --copy-generations 0 --joint-generations 1000 --population 64 --episodes-per-member 8 --fitness newcomer --workers 3 --eval-episodes 100 --out docs/results/add_specialist_pilot.md`.

## Verdict

- **Kill criterion:** KILL: most from-scratch newcomers under the frozen learned language did not reach the threshold within a budget of 38% of joint-retraining compute (median over runs, > 30%, so the true ratio is larger): the ecosystem needs a fixed typed core protocol.

### Pilot discussion (read this first)

**This is a partial pilot.** It used 2 source squads (R6 r = 3 runs 0 and 1, from the `r6-c-r3` GPU run's `learned_protocol_params.npz`), not the 3 planned. The box was shared with another job (load ~12 on 4 cores, ~4 s per generation instead of ~1), so the run-2 newcomers were stopped and left out. The joint reference is trained at population 64; the GPU job uses 128. With n = 2, every CI is wide and no Holm-corrected p can reach 0.05. Read the tables as a calibration, not as evidence.

What the pilot does show:

1. **A from-scratch newcomer is not cheap under any frozen protocol.**
   - The newcomers trained for 250 generations, which is 38% of the episodes the joint squad needed to reach its threshold.
   - On held-out episodes, newcomers into the learned language reached 1.0 food per episode, and newcomers into the hand-coded protocol 0.9.
   - The joint per-snake reference was 8.1–8.9, so the threshold was 7.8–8.0.
   - Over the same 250 generations, the jointly trained squad's per-snake food rose from 0.1 to 3.5. **Training everything from scratch learned faster per generation than training one newcomer into a frozen squad.** A shared network gets 4 snakes' experience per episode; the newcomer gets 1.
   - The kill criterion triggers even as a lower bound (> 38% > 30%). With 2 runs and a population-64 reference this is only a pilot signal; the GPU job decides it.
2. **Copying the shared specialist costs nothing.**
   - A newcomer initialised with a copy of the squad's own specialist weights works zero-shot (0 training generations): 10.4 food per episode with the learned language and 10.3 with the hand-coded protocol. That is above the joint reference (8.1) and close to the old snakes' per-snake food (~11–12).
   - It also *listens*: ablating only its channel drops it to ~1 food. Hand-coded: drop 9.1 [7.1, 11.1]. Learned: drop 9.3 [-0.7, 19.3], n = 2.
   - This works because R6's networks are colour- and receiver-agnostic. Adding a specialist of a kind the squad already has costs nothing, whatever the protocol.
3. **The frozen learned centre does address the newcomer.** The copy result above and the tests confirm it without retraining: permuting the snakes permutes the centre's codes, and adding snake 3 does not change what snakes 0–2 are told.
4. **The old specialists' food is roughly preserved.**
   - Their summed food is ~37 in the 3-snake squad, ~35 with a random newcomer (the extra body and food change the board), and 31.7–35.5 with a trained newcomer.
   - Every paired CI includes 0 at n = 2.
   - A competent copied newcomer costs the old snakes ~3–5 food (crowding) while adding ~10.
5. **Newcomer fitness has to be its own return.**
   - In a first attempt with the team return as fitness (R6's), a from-scratch newcomer stayed at ~0.06 food after 200 generations. That run was stopped and is not in these tables.
   - The cause: the frozen snakes' ~30 food per episode varies between ES members once the newcomer changes their trajectories, and that noise swamps the newcomer's signal.
   - Training on the newcomer's own return (food + survival) fixes this. It is the default (`--fitness newcomer`).

**Calibration for the GPU job.**

- **Measured here** (shared 4-core Xeon, 3 workers): one generation of population 64 × 8 episodes took ~1.0–1.3 s for the joint squad, and ~4 s for a newcomer under heavy load. Unloaded on this box, population 128 × 8 at N = 4 takes ~2.2 s, against ~1.5 s for R6's N = 3 (×1.45).
- **The `gpu` preset:** 10 runs; joint 1200 generations, scratch newcomers 800, copied newcomers 100; population 128 × 8; 16 workers. That is 39,000 generations in total.
- **Time estimate:** on the 20-core PC, R6 r = 3 measured ~4.3 s per generation with 20 parallel runs, so N = 4 should take ~6 s (×1.45). That gives ~65 worker-hours, ~4.1 h on 16 workers with the joint runs scheduled first, plus ~0.3 h of evaluation.
- **Why 800 scratch generations:** that is more than the ~650 generations the joint squad needed to reach its threshold here. A "never" in the GPU job therefore means a ratio > 100%, not a budget artefact.

## Design

- **Newcomer:** a 4th snake with a 4th colour joins a trained 3-snake squad (21×21, 2 food per colour, so the board gains 2 food items and a body). Only its network trains (`EvolutionStrategy` mask = 1 on its block); the old specialists' shared network and the centre stay frozen. Newcomer fitness = its own return (its food + 0.01 per step alive): with the team return the frozen snakes' ~30 food per episode varies between ES members once the newcomer perturbs their trajectories and swamps its first food items. Harm to the old snakes is measured below. The joint squad trains on the team return, as in R6.
- **The frozen centre addresses the newcomer unchanged:** both R6 networks are receiver-agnostic (the centre sees each receiver's own-colour food egocentrically; specialists see own/other planes, not colours), so with N = 4 the centre encodes colour 3's food for snake 3 in the same code (tested: permuting snakes permutes the codes).
- **Conditions:** (1) frozen learned centre + learned specialists; (2) frozen `HandCodedCentre` + its learned listeners; (3) joint retraining of all 4 snakes (one shared network, as in R6) + a learned centre from scratch; (4) silent squad. Newcomer init: `scratch` (random) or `copy` (the squad's own shared weights, a zero-shot clone because the network is colour-agnostic).
- **Compute metric:** reference = (3)'s per-snake food on the test episodes, per run; threshold = 90% of it. Compute to threshold = training episodes until the trailing 3-checkpoint mean of the validation curve reaches it (newcomer's food; per-snake mean for (3)). Ratio = newcomer / joint, run k vs run k; a newcomer that never reaches it counts as ∞. Kill if the median ratio of (1)-scratch > 30%.
- **Listening:** the harness's ablations applied to the newcomer's channel only (other receivers' messages untouched).

## Compute to reach the jointly trained newcomer's food

| condition | init | runs | reached threshold | median compute vs joint (to threshold) | runs within kill ratio | median gens to threshold | median gens to 90% of old snakes' food |
|---|---|---|---|---|---|---|---|
| learned | scratch | 2 | 0/2 | > 38% (budget spent) | 0% | never | never |
| learned | copy | 2 | 2/2 | 0% | 100% | 0 | never |
| hand-coded | scratch | 2 | 0/2 | > 38% (budget spent) | 0% | never | never |
| hand-coded | copy | 2 | 2/2 | 0% | 100% | 0 | never |
| joint | - | 2 | 2/2 | 100% | n/a | 665 | n/a |
| silent | scratch | 2 | 0/2 | > 38% (budget spent) | 0% | never | never |
| silent | copy | 2 | 0/2 | > 0% (budget spent) | 0% | never | never |

Per run:

| condition | init | run | ref food (joint, per snake) | threshold | gens to threshold | episodes | compute vs joint | vs joint full budget | wall-time vs joint |
|---|---|---|---|---|---|---|---|---|---|
| hand-coded | copy | 0 | 8.89 | 8.00 | 0 | 0 | 0% | 0% | 0% |
| hand-coded | copy | 1 | 8.68 | 7.81 | 0 | 0 | 0% | 0% | 0% |
| hand-coded | scratch | 0 | 8.89 | 8.00 | never | - | > 36% (budget spent) | never (∞) | never (∞) |
| hand-coded | scratch | 1 | 8.68 | 7.81 | never | - | > 40% (budget spent) | never (∞) | never (∞) |
| joint | - | 0 | 8.89 | 8.00 | 700 | 358,400 | n/a | n/a | n/a |
| joint | - | 1 | 8.68 | 7.81 | 630 | 322,560 | n/a | n/a | n/a |
| learned | copy | 0 | 8.89 | 8.00 | 0 | 0 | 0% | 0% | 0% |
| learned | copy | 1 | 8.68 | 7.81 | 0 | 0 | 0% | 0% | 0% |
| learned | scratch | 0 | 8.89 | 8.00 | never | - | > 36% (budget spent) | never (∞) | never (∞) |
| learned | scratch | 1 | 8.68 | 7.81 | never | - | > 40% (budget spent) | never (∞) | never (∞) |
| silent | copy | 0 | 8.89 | 8.00 | never | - | > 0% (budget spent) | never (∞) | never (∞) |
| silent | copy | 1 | 8.68 | 7.81 | never | - | > 0% (budget spent) | never (∞) | never (∞) |
| silent | scratch | 0 | 8.89 | 8.00 | never | - | > 36% (budget spent) | never (∞) | never (∞) |
| silent | scratch | 1 | 8.68 | 7.81 | never | - | > 40% (budget spent) | never (∞) | never (∞) |

## Newcomer food on held-out episodes

Mean ± 95% t-CI over runs. Score = the newcomer's (snake 3's) food per episode. Baselines put a scripted newcomer into run s's frozen learned squad.

| system | role | newcomer food | old snakes' food (sum of 3) | team food |
|---|---|---|---|---|
| (1) learned+newcomer[copy] | system | 10.43 ± 8.32 | 31.69 ± 16.96 | 42.12 ± 25.29 |
| (1) learned+newcomer[scratch] | system | 0.97 ± 2.54 | 34.71 ± 13.34 | 35.68 ± 10.80 |
| (2) hand-coded+newcomer[copy] | system | 10.32 ± 4.00 | 33.03 ± 9.21 | 43.35 ± 13.21 |
| (2) hand-coded+newcomer[scratch] | system | 0.88 ± 0.83 | 35.51 ± 12.58 | 36.38 ± 13.41 |
| (3) joint-retrain | system | 8.07 ± 1.14 | 27.08 ± 6.48 | 35.15 ± 5.34 |
| (4) silent+newcomer[copy] | system | 4.75 ± 3.11 | 12.77 ± 9.59 | 17.53 ± 6.48 |
| (4) silent+newcomer[scratch] | system | 0.33 ± 0.06 | 13.07 ± 8.70 | 13.40 ± 8.64 |
| newcomer:random | baseline | 0.12 ± 0.25 | 35.25 ± 11.50 | 35.36 ± 11.24 |
| newcomer:no-comm-greedy | baseline | 2.63 ± 1.46 | 33.84 ± 20.77 | 36.48 ± 22.24 |
| newcomer:oracle | baseline | 15.91 ± 2.22 | 27.31 ± 11.56 | 43.23 ± 9.34 |

Comparisons vs baselines (Mann-Whitney over runs, Holm):

```
Task: squad-add-specialist-r3  (seeds: 2)
Baselines: newcomer:random, newcomer:no-comm-greedy, newcomer:oracle

condition                         role      n  mean ± 95% CI   IQM [95% CI]           
--------------------------------  --------  -  --------------  -----------------------
(1) learned+newcomer[copy]        system    2  10.435 ± 8.323  10.435 [9.780, 11.090] 
(1) learned+newcomer[scratch]     system    2  0.970 ± 2.541   0.970 [0.770, 1.170]   
(2) hand-coded+newcomer[copy]     system    2  10.315 ± 4.002  10.315 [10.000, 10.630]
(2) hand-coded+newcomer[scratch]  system    2  0.875 ± 0.826   0.875 [0.810, 0.940]   
(3) joint-retrain                 system    2  8.070 ± 1.144   8.070 [7.980, 8.160]   
(4) silent+newcomer[copy]         system    2  4.755 ± 3.113   4.755 [4.510, 5.000]   
(4) silent+newcomer[scratch]      system    2  0.325 ± 0.064   0.325 [0.320, 0.330]   
newcomer:random                   baseline  2  0.120 ± 0.254   0.120 [0.100, 0.140]   
newcomer:no-comm-greedy           baseline  2  2.635 ± 1.461   2.635 [2.520, 2.750]   
newcomer:oracle                   baseline  2  15.915 ± 2.224  15.915 [15.740, 16.090]

system                            intervention    ablated mean  drop [95% CI]           p (Holm)  A12    listening
--------------------------------  --------------  ------------  ----------------------  --------  -----  ---------
(1) learned+newcomer[copy]        zero            1.145         9.290 [-0.748, 19.328]  1.000     1.000  no       
(1) learned+newcomer[copy]        shuffle_batch   0.790         9.645 [-0.711, 20.001]  1.000     1.000  no       
(1) learned+newcomer[copy]        random_matched  0.705         9.730 [-0.562, 20.022]  1.000     1.000  no       
(1) learned+newcomer[scratch]     zero            0.270         0.700 [-1.460, 2.860]   1.000     1.000  no       
(1) learned+newcomer[scratch]     shuffle_batch   0.210         0.760 [-2.162, 3.682]   1.000     1.000  no       
(1) learned+newcomer[scratch]     random_matched  0.235         0.735 [-2.124, 3.594]   1.000     1.000  no       
(2) hand-coded+newcomer[copy]     zero            1.220         9.095 [7.126, 11.064]   1.000     1.000  yes      
(2) hand-coded+newcomer[copy]     shuffle_batch   0.785         9.530 [6.481, 12.579]   1.000     1.000  yes      
(2) hand-coded+newcomer[copy]     random_matched  0.835         9.480 [6.303, 12.657]   1.000     1.000  yes      
(2) hand-coded+newcomer[scratch]  zero            0.230         0.645 [0.073, 1.217]    1.000     1.000  yes      
(2) hand-coded+newcomer[scratch]  shuffle_batch   0.230         0.645 [-0.816, 2.106]   1.000     1.000  no       
(2) hand-coded+newcomer[scratch]  random_matched  0.175         0.700 [0.065, 1.335]    1.000     1.000  yes      
(3) joint-retrain                 zero            0.815         7.255 [5.413, 9.097]    1.000     1.000  yes      
(3) joint-retrain                 shuffle_batch   0.640         7.430 [5.270, 9.590]    1.000     1.000  yes      
(3) joint-retrain                 random_matched  0.710         7.360 [3.675, 11.045]   1.000     1.000  yes      
(4) silent+newcomer[copy]         zero            4.755         0.000 [0.000, 0.000]    1.000     0.500  no       
(4) silent+newcomer[copy]         shuffle_batch   4.755         0.000 [0.000, 0.000]    1.000     0.500  no       
(4) silent+newcomer[copy]         random_matched  4.755         0.000 [0.000, 0.000]    1.000     0.500  no       
(4) silent+newcomer[scratch]      zero            0.325         0.000 [0.000, 0.000]    1.000     0.500  no       
(4) silent+newcomer[scratch]      shuffle_batch   0.325         0.000 [0.000, 0.000]    1.000     0.500  no       
(4) silent+newcomer[scratch]      random_matched  0.325         0.000 [0.000, 0.000]    1.000     0.500  no       

comparison                                                   mean diff  p      p (Holm)  A12  
-----------------------------------------------------------  ---------  -----  --------  -----
(1) learned+newcomer[copy] vs newcomer:random                10.315     0.333  1.000     1.000
(1) learned+newcomer[copy] vs newcomer:no-comm-greedy        7.800      0.333  1.000     1.000
(1) learned+newcomer[copy] vs newcomer:oracle                -5.480     0.333  1.000     0.000
(1) learned+newcomer[scratch] vs newcomer:random             0.850      0.333  1.000     1.000
(1) learned+newcomer[scratch] vs newcomer:no-comm-greedy     -1.665     0.333  1.000     0.000
(1) learned+newcomer[scratch] vs newcomer:oracle             -14.945    0.333  1.000     0.000
(2) hand-coded+newcomer[copy] vs newcomer:random             10.195     0.333  1.000     1.000
(2) hand-coded+newcomer[copy] vs newcomer:no-comm-greedy     7.680      0.333  1.000     1.000
(2) hand-coded+newcomer[copy] vs newcomer:oracle             -5.600     0.333  1.000     0.000
(2) hand-coded+newcomer[scratch] vs newcomer:random          0.755      0.333  1.000     1.000
(2) hand-coded+newcomer[scratch] vs newcomer:no-comm-greedy  -1.760     0.333  1.000     0.000
(2) hand-coded+newcomer[scratch] vs newcomer:oracle          -15.040    0.333  1.000     0.000
(3) joint-retrain vs newcomer:random                         7.950      0.333  1.000     1.000
(3) joint-retrain vs newcomer:no-comm-greedy                 5.435      0.333  1.000     1.000
(3) joint-retrain vs newcomer:oracle                         -7.845     0.333  1.000     0.000
(4) silent+newcomer[copy] vs newcomer:random                 4.635      0.333  1.000     1.000
(4) silent+newcomer[copy] vs newcomer:no-comm-greedy         2.120      0.333  1.000     1.000
(4) silent+newcomer[copy] vs newcomer:oracle                 -11.160    0.333  1.000     0.000
(4) silent+newcomer[scratch] vs newcomer:random              0.205      0.333  1.000     1.000
(4) silent+newcomer[scratch] vs newcomer:no-comm-greedy      -2.310     0.333  1.000     0.000
(4) silent+newcomer[scratch] vs newcomer:oracle              -15.590    0.333  1.000     0.000
```

## Does the newcomer listen? (ablations of its channel)

Drop = intact − ablated newcomer food, paired by run; p = Wilcoxon, Holm per system.

| system | intervention (newcomer's channel only) | intact | ablated | drop [95% CI] | p (Holm) | listening |
|---|---|---|---|---|---|---|
| (1) learned+newcomer[copy] | zero | 10.43 | 1.15 | 9.29 [-0.75, 19.33] | 1.000 | no |
| (1) learned+newcomer[copy] | shuffle_batch | 10.43 | 0.79 | 9.64 [-0.71, 20.00] | 1.000 | no |
| (1) learned+newcomer[copy] | random_matched | 10.43 | 0.71 | 9.73 [-0.56, 20.02] | 1.000 | no |
| (1) learned+newcomer[scratch] | zero | 0.97 | 0.27 | 0.70 [-1.46, 2.86] | 1.000 | no |
| (1) learned+newcomer[scratch] | shuffle_batch | 0.97 | 0.21 | 0.76 [-2.16, 3.68] | 1.000 | no |
| (1) learned+newcomer[scratch] | random_matched | 0.97 | 0.23 | 0.73 [-2.12, 3.59] | 1.000 | no |
| (2) hand-coded+newcomer[copy] | zero | 10.32 | 1.22 | 9.09 [7.13, 11.06] | 1.000 | yes |
| (2) hand-coded+newcomer[copy] | shuffle_batch | 10.32 | 0.78 | 9.53 [6.48, 12.58] | 1.000 | yes |
| (2) hand-coded+newcomer[copy] | random_matched | 10.32 | 0.83 | 9.48 [6.30, 12.66] | 1.000 | yes |
| (2) hand-coded+newcomer[scratch] | zero | 0.88 | 0.23 | 0.65 [0.07, 1.22] | 1.000 | yes |
| (2) hand-coded+newcomer[scratch] | shuffle_batch | 0.88 | 0.23 | 0.65 [-0.82, 2.11] | 1.000 | no |
| (2) hand-coded+newcomer[scratch] | random_matched | 0.88 | 0.17 | 0.70 [0.06, 1.34] | 1.000 | yes |
| (3) joint-retrain | zero | 8.07 | 0.81 | 7.26 [5.41, 9.10] | 1.000 | yes |
| (3) joint-retrain | shuffle_batch | 8.07 | 0.64 | 7.43 [5.27, 9.59] | 1.000 | yes |
| (3) joint-retrain | random_matched | 8.07 | 0.71 | 7.36 [3.68, 11.04] | 1.000 | yes |
| (4) silent+newcomer[copy] | zero | 4.75 | 4.75 | 0.00 [0.00, 0.00] | 1.000 | no |
| (4) silent+newcomer[copy] | shuffle_batch | 4.75 | 4.75 | 0.00 [0.00, 0.00] | 1.000 | no |
| (4) silent+newcomer[copy] | random_matched | 4.75 | 4.75 | 0.00 [0.00, 0.00] | 1.000 | no |
| (4) silent+newcomer[scratch] | zero | 0.33 | 0.33 | 0.00 [0.00, 0.00] | 1.000 | no |
| (4) silent+newcomer[scratch] | shuffle_batch | 0.33 | 0.33 | 0.00 [0.00, 0.00] | 1.000 | no |
| (4) silent+newcomer[scratch] | random_matched | 0.33 | 0.33 | 0.00 [0.00, 0.00] | 1.000 | no |

## Are the old specialists' food preserved?

The three old snakes' summed food per held-out episode, mean ± 95% CI over runs; differences paired by run (t-CI, Wilcoxon p).

| source squad | newcomer init | 3-snake squad | 4 snakes, random newcomer | 4 snakes, trained newcomer | trained − 3-snake [95% CI] (p) | trained − random newcomer [95% CI] (p) |
|---|---|---|---|---|---|---|
| learned | copy | 37.19 ± 9.53 | 35.25 ± 11.50 | 31.69 ± 16.96 | -5.50 ± 7.43 (0.500) | -3.56 ± 5.46 (0.500) |
| learned | scratch | 37.19 ± 9.53 | 35.25 ± 11.50 | 34.71 ± 13.34 | -2.48 ± 3.81 (0.500) | -0.54 ± 1.84 (0.500) |
| hand-coded | copy | 38.50 ± 4.19 | 36.03 ± 5.72 | 33.03 ± 9.21 | -5.46 ± 5.02 (0.500) | -2.99 ± 3.49 (0.500) |
| hand-coded | scratch | 38.50 ± 4.19 | 36.03 ± 5.72 | 35.51 ± 12.58 | -2.99 ± 8.39 (0.500) | -0.52 ± 6.86 (1.000) |
| silent | copy | 14.91 ± 6.35 | 13.72 ± 7.18 | 12.77 ± 9.59 | -2.14 ± 15.95 (0.500) | -0.95 ± 2.41 (0.500) |
| silent | scratch | 14.91 ± 6.35 | 13.72 ± 7.18 | 13.07 ± 8.70 | -1.83 ± 15.06 (0.500) | -0.65 ± 1.52 (0.500) |

## Learning curves

Validation food (mean over runs) at quantiles of each run's curve (median generation in brackets). Full curves in `add_specialist_pilot_curves.csv`.

| condition | init | metric | q0 | q10 | q25 | q50 | q75 | q100 |
|---|---|---|---|---|---|---|---|---|
| learned | scratch | val_newcomer_food | 0.09 (g0) | 0.10 (g20) | 0.30 (g60) | 0.79 (g120) | 0.98 (g190) | 1.17 (g250) |
| learned | copy | val_newcomer_food | 10.72 (g0) | 10.72 (g0) | 10.72 (g0) | 10.72 (g0) | 10.72 (g0) | 10.72 (g0) |
| hand-coded | scratch | val_newcomer_food | 0.06 (g0) | 0.14 (g20) | 0.12 (g60) | 0.15 (g120) | 0.52 (g190) | 0.86 (g250) |
| hand-coded | copy | val_newcomer_food | 9.83 (g0) | 9.83 (g0) | 9.83 (g0) | 9.83 (g0) | 9.83 (g0) | 9.83 (g0) |
| joint | - | val_per_snake_food | 0.09 (g0) | 0.46 (g100) | 3.51 (g250) | 7.02 (g500) | 8.25 (g750) | 8.63 (g1000) |
| silent | scratch | val_newcomer_food | 0.10 (g0) | 0.14 (g20) | 0.17 (g60) | 0.30 (g120) | 0.41 (g190) | 0.26 (g250) |
| silent | copy | val_newcomer_food | 4.81 (g0) | 4.81 (g0) | 4.81 (g0) | 4.81 (g0) | 4.81 (g0) | 4.81 (g0) |
