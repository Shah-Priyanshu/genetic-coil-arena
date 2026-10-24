# Squad: can a new specialist join a learned language? (R7)

- Date: 2026-10-23
- Commit: 4def4e7 (code; this report is committed on top of it)
- Compute: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, 28 logical cores, 16 worker processes. Training wall time 251.6 min (64.10 CPU-hours), evaluation 2.0 min.
- Source squads: R6 r = 3 runs 0–9 (`git:gpu-results:runs/r6-c-r3/artifacts/learned_protocol_params.npz`, sha256 fb60873dacba…): 3 snakes, one shared specialist network per squad, trained 2000 ES generations × population 128 × 8 episodes.
- Training: ES population 128 × 8 episodes per member; generations: newcomer from scratch 800, newcomer copied 100, joint retraining 1200. Validation every 10 generations on 64 episodes.
- Held-out evaluation: 200 test episodes per run (R6's test seeds, now with 4 snakes). Runs are the independent units for every CI/test.

Reproduce with `python scripts/add_specialist.py --scale gpu --runs 10 --generations 800 --copy-generations 100 --joint-generations 1200 --population 128 --episodes-per-member 8 --fitness newcomer --workers 16 --eval-episodes 200 --out docs/results/add_specialist.md`.

## Verdict

- **Kill criterion:** KILL: most from-scratch newcomers under the frozen learned language did not reach the threshold within a budget of 124% of joint-retraining compute (median over runs, > 30%, so the true ratio is larger): the ecosystem needs a fixed typed core protocol.

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
| learned | scratch | 10 | 0/10 | > 124% (budget spent) | 0% | never | never |
| learned | copy | 10 | 9/10 | 0% | 90% | 0 | 85 |
| hand-coded | scratch | 10 | 1/10 | > 116% (budget spent) | 0% | never | never |
| hand-coded | copy | 10 | 9/10 | 0% | 90% | 0 | never |
| joint | - | 10 | 10/10 | 100% | n/a | 645 | n/a |
| silent | scratch | 10 | 0/10 | > 124% (budget spent) | 0% | never | never |
| silent | copy | 10 | 0/10 | > 16% (budget spent) | 0% | never | never |

Per run:

| condition | init | run | ref food (joint, per snake) | threshold | gens to threshold | episodes | compute vs joint | vs joint full budget | wall-time vs joint |
|---|---|---|---|---|---|---|---|---|---|
| hand-coded | copy | 0 | 10.84 | 9.75 | 10 | 10,240 | 1% | 1% | 1% |
| hand-coded | copy | 1 | 9.44 | 8.49 | 0 | 0 | 0% | 0% | 0% |
| hand-coded | copy | 2 | 10.05 | 9.04 | 0 | 0 | 0% | 0% | 0% |
| hand-coded | copy | 3 | 10.55 | 9.50 | 0 | 0 | 0% | 0% | 0% |
| hand-coded | copy | 4 | 10.69 | 9.62 | 10 | 10,240 | 2% | 1% | 2% |
| hand-coded | copy | 5 | 10.14 | 9.13 | 0 | 0 | 0% | 0% | 0% |
| hand-coded | copy | 6 | 10.02 | 9.02 | never | - | > 32% (budget spent) | never (∞) | never (∞) |
| hand-coded | copy | 7 | 9.95 | 8.95 | 0 | 0 | 0% | 0% | 0% |
| hand-coded | copy | 8 | 7.67 | 6.90 | 0 | 0 | 0% | 0% | 0% |
| hand-coded | copy | 9 | 10.79 | 9.71 | 0 | 0 | 0% | 0% | 0% |
| hand-coded | scratch | 0 | 10.84 | 9.75 | never | - | > 90% (budget spent) | never (∞) | never (∞) |
| hand-coded | scratch | 1 | 9.44 | 8.49 | never | - | > 108% (budget spent) | never (∞) | never (∞) |
| hand-coded | scratch | 2 | 10.05 | 9.04 | never | - | > 103% (budget spent) | never (∞) | never (∞) |
| hand-coded | scratch | 3 | 10.55 | 9.50 | never | - | > 105% (budget spent) | never (∞) | never (∞) |
| hand-coded | scratch | 4 | 10.69 | 9.62 | never | - | > 160% (budget spent) | never (∞) | never (∞) |
| hand-coded | scratch | 5 | 10.14 | 9.13 | never | - | > 157% (budget spent) | never (∞) | never (∞) |
| hand-coded | scratch | 6 | 10.02 | 9.02 | never | - | > 258% (budget spent) | never (∞) | never (∞) |
| hand-coded | scratch | 7 | 9.95 | 8.95 | never | - | > 195% (budget spent) | never (∞) | never (∞) |
| hand-coded | scratch | 8 | 7.67 | 6.90 | 590 | 604,160 | 92% | 49% | 92% |
| hand-coded | scratch | 9 | 10.79 | 9.71 | never | - | > 123% (budget spent) | never (∞) | never (∞) |
| joint | - | 0 | 10.84 | 9.75 | 890 | 911,360 | n/a | n/a | n/a |
| joint | - | 1 | 9.44 | 8.49 | 740 | 757,760 | n/a | n/a | n/a |
| joint | - | 2 | 10.05 | 9.04 | 780 | 798,720 | n/a | n/a | n/a |
| joint | - | 3 | 10.55 | 9.50 | 760 | 778,240 | n/a | n/a | n/a |
| joint | - | 4 | 10.69 | 9.62 | 500 | 512,000 | n/a | n/a | n/a |
| joint | - | 5 | 10.14 | 9.13 | 510 | 522,240 | n/a | n/a | n/a |
| joint | - | 6 | 10.02 | 9.02 | 310 | 317,440 | n/a | n/a | n/a |
| joint | - | 7 | 9.95 | 8.95 | 410 | 419,840 | n/a | n/a | n/a |
| joint | - | 8 | 7.67 | 6.90 | 640 | 655,360 | n/a | n/a | n/a |
| joint | - | 9 | 10.79 | 9.71 | 650 | 665,600 | n/a | n/a | n/a |
| learned | copy | 0 | 10.84 | 9.75 | 10 | 10,240 | 1% | 1% | 1% |
| learned | copy | 1 | 9.44 | 8.49 | 0 | 0 | 0% | 0% | 0% |
| learned | copy | 2 | 10.05 | 9.04 | 0 | 0 | 0% | 0% | 0% |
| learned | copy | 3 | 10.55 | 9.50 | 0 | 0 | 0% | 0% | 0% |
| learned | copy | 4 | 10.69 | 9.62 | 0 | 0 | 0% | 0% | 0% |
| learned | copy | 5 | 10.14 | 9.13 | 0 | 0 | 0% | 0% | 0% |
| learned | copy | 6 | 10.02 | 9.02 | never | - | > 32% (budget spent) | never (∞) | never (∞) |
| learned | copy | 7 | 9.95 | 8.95 | 0 | 0 | 0% | 0% | 0% |
| learned | copy | 8 | 7.67 | 6.90 | 0 | 0 | 0% | 0% | 0% |
| learned | copy | 9 | 10.79 | 9.71 | 10 | 10,240 | 2% | 1% | 2% |
| learned | scratch | 0 | 10.84 | 9.75 | never | - | > 90% (budget spent) | never (∞) | never (∞) |
| learned | scratch | 1 | 9.44 | 8.49 | never | - | > 108% (budget spent) | never (∞) | never (∞) |
| learned | scratch | 2 | 10.05 | 9.04 | never | - | > 103% (budget spent) | never (∞) | never (∞) |
| learned | scratch | 3 | 10.55 | 9.50 | never | - | > 105% (budget spent) | never (∞) | never (∞) |
| learned | scratch | 4 | 10.69 | 9.62 | never | - | > 160% (budget spent) | never (∞) | never (∞) |
| learned | scratch | 5 | 10.14 | 9.13 | never | - | > 157% (budget spent) | never (∞) | never (∞) |
| learned | scratch | 6 | 10.02 | 9.02 | never | - | > 258% (budget spent) | never (∞) | never (∞) |
| learned | scratch | 7 | 9.95 | 8.95 | never | - | > 195% (budget spent) | never (∞) | never (∞) |
| learned | scratch | 8 | 7.67 | 6.90 | never | - | > 125% (budget spent) | never (∞) | never (∞) |
| learned | scratch | 9 | 10.79 | 9.71 | never | - | > 123% (budget spent) | never (∞) | never (∞) |
| silent | copy | 0 | 10.84 | 9.75 | never | - | > 11% (budget spent) | never (∞) | never (∞) |
| silent | copy | 1 | 9.44 | 8.49 | never | - | > 14% (budget spent) | never (∞) | never (∞) |
| silent | copy | 2 | 10.05 | 9.04 | never | - | > 13% (budget spent) | never (∞) | never (∞) |
| silent | copy | 3 | 10.55 | 9.50 | never | - | > 13% (budget spent) | never (∞) | never (∞) |
| silent | copy | 4 | 10.69 | 9.62 | never | - | > 20% (budget spent) | never (∞) | never (∞) |
| silent | copy | 5 | 10.14 | 9.13 | never | - | > 20% (budget spent) | never (∞) | never (∞) |
| silent | copy | 6 | 10.02 | 9.02 | never | - | > 32% (budget spent) | never (∞) | never (∞) |
| silent | copy | 7 | 9.95 | 8.95 | never | - | > 24% (budget spent) | never (∞) | never (∞) |
| silent | copy | 8 | 7.67 | 6.90 | never | - | > 16% (budget spent) | never (∞) | never (∞) |
| silent | copy | 9 | 10.79 | 9.71 | never | - | > 15% (budget spent) | never (∞) | never (∞) |
| silent | scratch | 0 | 10.84 | 9.75 | never | - | > 90% (budget spent) | never (∞) | never (∞) |
| silent | scratch | 1 | 9.44 | 8.49 | never | - | > 108% (budget spent) | never (∞) | never (∞) |
| silent | scratch | 2 | 10.05 | 9.04 | never | - | > 103% (budget spent) | never (∞) | never (∞) |
| silent | scratch | 3 | 10.55 | 9.50 | never | - | > 105% (budget spent) | never (∞) | never (∞) |
| silent | scratch | 4 | 10.69 | 9.62 | never | - | > 160% (budget spent) | never (∞) | never (∞) |
| silent | scratch | 5 | 10.14 | 9.13 | never | - | > 157% (budget spent) | never (∞) | never (∞) |
| silent | scratch | 6 | 10.02 | 9.02 | never | - | > 258% (budget spent) | never (∞) | never (∞) |
| silent | scratch | 7 | 9.95 | 8.95 | never | - | > 195% (budget spent) | never (∞) | never (∞) |
| silent | scratch | 8 | 7.67 | 6.90 | never | - | > 125% (budget spent) | never (∞) | never (∞) |
| silent | scratch | 9 | 10.79 | 9.71 | never | - | > 123% (budget spent) | never (∞) | never (∞) |

## Newcomer food on held-out episodes

Mean ± 95% t-CI over runs. Score = the newcomer's (snake 3's) food per episode. Baselines put a scripted newcomer into run s's frozen learned squad.

| system | role | newcomer food | old snakes' food (sum of 3) | team food |
|---|---|---|---|---|
| (1) learned+newcomer[copy] | system | 9.91 ± 0.68 | 30.92 ± 1.58 | 40.83 ± 2.13 |
| (1) learned+newcomer[scratch] | system | 5.74 ± 1.76 | 31.86 ± 1.77 | 37.60 ± 1.83 |
| (2) hand-coded+newcomer[copy] | system | 9.61 ± 0.54 | 29.81 ± 2.15 | 39.42 ± 2.61 |
| (2) hand-coded+newcomer[scratch] | system | 5.26 ± 1.56 | 30.82 ± 2.51 | 36.08 ± 1.96 |
| (3) joint-retrain | system | 9.85 ± 0.59 | 30.21 ± 2.21 | 40.06 ± 2.68 |
| (4) silent+newcomer[copy] | system | 4.07 ± 0.35 | 13.07 ± 1.03 | 17.14 ± 1.25 |
| (4) silent+newcomer[scratch] | system | 1.20 ± 0.17 | 12.66 ± 0.70 | 13.87 ± 0.74 |
| newcomer:random | baseline | 0.08 ± 0.02 | 34.52 ± 1.54 | 34.60 ± 1.52 |
| newcomer:no-comm-greedy | baseline | 2.38 ± 0.13 | 33.00 ± 1.68 | 35.38 ± 1.63 |
| newcomer:oracle | baseline | 17.50 ± 0.49 | 27.01 ± 1.58 | 44.51 ± 1.36 |

Comparisons vs baselines (Mann-Whitney over runs, Holm):

```
Task: squad-add-specialist-r3  (seeds: 10)
Baselines: newcomer:random, newcomer:no-comm-greedy, newcomer:oracle

condition                         role      n   mean ± 95% CI   IQM [95% CI]           
--------------------------------  --------  --  --------------  -----------------------
(1) learned+newcomer[copy]        system    10  9.911 ± 0.683   10.019 [9.179, 10.673] 
(1) learned+newcomer[scratch]     system    10  5.743 ± 1.757   5.840 [3.800, 7.658]   
(2) hand-coded+newcomer[copy]     system    10  9.611 ± 0.542   9.691 [9.150, 10.103]  
(2) hand-coded+newcomer[scratch]  system    10  5.261 ± 1.564   5.353 [3.755, 6.857]   
(3) joint-retrain                 system    10  9.846 ± 0.587   9.938 [9.310, 10.457]  
(4) silent+newcomer[copy]         system    10  4.074 ± 0.354   4.198 [3.758, 4.392]   
(4) silent+newcomer[scratch]      system    10  1.204 ± 0.172   1.196 [1.040, 1.373]   
newcomer:random                   baseline  10  0.084 ± 0.020   0.083 [0.058, 0.108]   
newcomer:no-comm-greedy           baseline  10  2.378 ± 0.129   2.381 [2.252, 2.507]   
newcomer:oracle                   baseline  10  17.498 ± 0.490  17.618 [16.972, 17.994]

system                            intervention    ablated mean  drop [95% CI]         p (Holm)  A12    listening
--------------------------------  --------------  ------------  --------------------  --------  -----  ---------
(1) learned+newcomer[copy]        zero            1.026         8.884 [8.118, 9.651]  0.006     1.000  yes      
(1) learned+newcomer[copy]        shuffle_batch   0.807         9.103 [8.434, 9.773]  0.006     1.000  yes      
(1) learned+newcomer[copy]        random_matched  0.848         9.064 [8.371, 9.756]  0.006     1.000  yes      
(1) learned+newcomer[scratch]     zero            0.498         5.246 [3.638, 6.853]  0.006     1.000  yes      
(1) learned+newcomer[scratch]     shuffle_batch   0.391         5.352 [3.686, 7.018]  0.006     1.000  yes      
(1) learned+newcomer[scratch]     random_matched  0.371         5.372 [3.680, 7.064]  0.006     1.000  yes      
(2) hand-coded+newcomer[copy]     zero            1.167         8.443 [7.956, 8.930]  0.006     1.000  yes      
(2) hand-coded+newcomer[copy]     shuffle_batch   0.693         8.918 [8.440, 9.395]  0.006     1.000  yes      
(2) hand-coded+newcomer[copy]     random_matched  0.710         8.900 [8.410, 9.391]  0.006     1.000  yes      
(2) hand-coded+newcomer[scratch]  zero            0.657         4.605 [3.112, 6.097]  0.006     1.000  yes      
(2) hand-coded+newcomer[scratch]  shuffle_batch   0.429         4.832 [3.339, 6.325]  0.006     1.000  yes      
(2) hand-coded+newcomer[scratch]  random_matched  0.460         4.801 [3.324, 6.279]  0.006     1.000  yes      
(3) joint-retrain                 zero            1.228         8.617 [8.047, 9.187]  0.006     1.000  yes      
(3) joint-retrain                 shuffle_batch   0.774         9.071 [8.499, 9.643]  0.006     1.000  yes      
(3) joint-retrain                 random_matched  0.844         9.002 [8.408, 9.596]  0.006     1.000  yes      
(4) silent+newcomer[copy]         zero            4.074         0.000 [0.000, 0.000]  1.000     0.500  no       
(4) silent+newcomer[copy]         shuffle_batch   4.074         0.000 [0.000, 0.000]  1.000     0.500  no       
(4) silent+newcomer[copy]         random_matched  4.074         0.000 [0.000, 0.000]  1.000     0.500  no       
(4) silent+newcomer[scratch]      zero            1.204         0.000 [0.000, 0.000]  1.000     0.500  no       
(4) silent+newcomer[scratch]      shuffle_batch   1.204         0.000 [0.000, 0.000]  1.000     0.500  no       
(4) silent+newcomer[scratch]      random_matched  1.204         0.000 [0.000, 0.000]  1.000     0.500  no       

comparison                                                   mean diff  p      p (Holm)  A12  
-----------------------------------------------------------  ---------  -----  --------  -----
(1) learned+newcomer[copy] vs newcomer:random                9.827      0.000  0.004     1.000
(1) learned+newcomer[copy] vs newcomer:no-comm-greedy        7.532      0.000  0.004     1.000
(1) learned+newcomer[copy] vs newcomer:oracle                -7.587     0.000  0.004     0.000
(1) learned+newcomer[scratch] vs newcomer:random             5.659      0.000  0.004     1.000
(1) learned+newcomer[scratch] vs newcomer:no-comm-greedy     3.365      0.002  0.004     0.920
(1) learned+newcomer[scratch] vs newcomer:oracle             -11.755    0.000  0.004     0.000
(2) hand-coded+newcomer[copy] vs newcomer:random             9.527      0.000  0.004     1.000
(2) hand-coded+newcomer[copy] vs newcomer:no-comm-greedy     7.232      0.000  0.004     1.000
(2) hand-coded+newcomer[copy] vs newcomer:oracle             -7.887     0.000  0.004     0.000
(2) hand-coded+newcomer[scratch] vs newcomer:random          5.178      0.000  0.004     1.000
(2) hand-coded+newcomer[scratch] vs newcomer:no-comm-greedy  2.883      0.003  0.004     0.900
(2) hand-coded+newcomer[scratch] vs newcomer:oracle          -12.236    0.000  0.004     0.000
(3) joint-retrain vs newcomer:random                         9.762      0.000  0.004     1.000
(3) joint-retrain vs newcomer:no-comm-greedy                 7.467      0.000  0.004     1.000
(3) joint-retrain vs newcomer:oracle                         -7.652     0.000  0.004     0.000
(4) silent+newcomer[copy] vs newcomer:random                 3.990      0.000  0.004     1.000
(4) silent+newcomer[copy] vs newcomer:no-comm-greedy         1.696      0.000  0.004     1.000
(4) silent+newcomer[copy] vs newcomer:oracle                 -13.424    0.000  0.004     0.000
(4) silent+newcomer[scratch] vs newcomer:random              1.120      0.000  0.004     1.000
(4) silent+newcomer[scratch] vs newcomer:no-comm-greedy      -1.174     0.000  0.004     0.000
(4) silent+newcomer[scratch] vs newcomer:oracle              -16.294    0.000  0.004     0.000
```

## Does the newcomer listen? (ablations of its channel)

Drop = intact − ablated newcomer food, paired by run; p = Wilcoxon, Holm per system.

| system | intervention (newcomer's channel only) | intact | ablated | drop [95% CI] | p (Holm) | listening |
|---|---|---|---|---|---|---|
| (1) learned+newcomer[copy] | zero | 9.91 | 1.03 | 8.88 [8.12, 9.65] | 0.0059 | yes |
| (1) learned+newcomer[copy] | shuffle_batch | 9.91 | 0.81 | 9.10 [8.43, 9.77] | 0.0059 | yes |
| (1) learned+newcomer[copy] | random_matched | 9.91 | 0.85 | 9.06 [8.37, 9.76] | 0.0059 | yes |
| (1) learned+newcomer[scratch] | zero | 5.74 | 0.50 | 5.25 [3.64, 6.85] | 0.0059 | yes |
| (1) learned+newcomer[scratch] | shuffle_batch | 5.74 | 0.39 | 5.35 [3.69, 7.02] | 0.0059 | yes |
| (1) learned+newcomer[scratch] | random_matched | 5.74 | 0.37 | 5.37 [3.68, 7.06] | 0.0059 | yes |
| (2) hand-coded+newcomer[copy] | zero | 9.61 | 1.17 | 8.44 [7.96, 8.93] | 0.0059 | yes |
| (2) hand-coded+newcomer[copy] | shuffle_batch | 9.61 | 0.69 | 8.92 [8.44, 9.40] | 0.0059 | yes |
| (2) hand-coded+newcomer[copy] | random_matched | 9.61 | 0.71 | 8.90 [8.41, 9.39] | 0.0059 | yes |
| (2) hand-coded+newcomer[scratch] | zero | 5.26 | 0.66 | 4.60 [3.11, 6.10] | 0.0059 | yes |
| (2) hand-coded+newcomer[scratch] | shuffle_batch | 5.26 | 0.43 | 4.83 [3.34, 6.33] | 0.0059 | yes |
| (2) hand-coded+newcomer[scratch] | random_matched | 5.26 | 0.46 | 4.80 [3.32, 6.28] | 0.0059 | yes |
| (3) joint-retrain | zero | 9.85 | 1.23 | 8.62 [8.05, 9.19] | 0.0059 | yes |
| (3) joint-retrain | shuffle_batch | 9.85 | 0.77 | 9.07 [8.50, 9.64] | 0.0059 | yes |
| (3) joint-retrain | random_matched | 9.85 | 0.84 | 9.00 [8.41, 9.60] | 0.0059 | yes |
| (4) silent+newcomer[copy] | zero | 4.07 | 4.07 | 0.00 [0.00, 0.00] | 1.000 | no |
| (4) silent+newcomer[copy] | shuffle_batch | 4.07 | 4.07 | 0.00 [0.00, 0.00] | 1.000 | no |
| (4) silent+newcomer[copy] | random_matched | 4.07 | 4.07 | 0.00 [0.00, 0.00] | 1.000 | no |
| (4) silent+newcomer[scratch] | zero | 1.20 | 1.20 | 0.00 [0.00, 0.00] | 1.000 | no |
| (4) silent+newcomer[scratch] | shuffle_batch | 1.20 | 1.20 | 0.00 [0.00, 0.00] | 1.000 | no |
| (4) silent+newcomer[scratch] | random_matched | 1.20 | 1.20 | 0.00 [0.00, 0.00] | 1.000 | no |

## Are the old specialists' food preserved?

The three old snakes' summed food per held-out episode, mean ± 95% CI over runs; differences paired by run (t-CI, Wilcoxon p).

| source squad | newcomer init | 3-snake squad | 4 snakes, random newcomer | 4 snakes, trained newcomer | trained − 3-snake [95% CI] (p) | trained − random newcomer [95% CI] (p) |
|---|---|---|---|---|---|---|
| learned | copy | 35.43 ± 1.59 | 34.52 ± 1.54 | 30.92 ± 1.58 | -4.51 ± 0.35 (0.002) | -3.59 ± 0.37 (0.002) |
| learned | scratch | 35.43 ± 1.59 | 34.52 ± 1.54 | 31.86 ± 1.77 | -3.57 ± 0.86 (0.002) | -2.65 ± 0.83 (0.002) |
| hand-coded | copy | 34.89 ± 2.37 | 33.58 ± 2.16 | 29.81 ± 2.15 | -5.08 ± 0.63 (0.002) | -3.78 ± 0.75 (0.002) |
| hand-coded | scratch | 34.89 ± 2.37 | 33.58 ± 2.16 | 30.82 ± 2.51 | -4.07 ± 0.78 (0.002) | -2.76 ± 0.86 (0.002) |
| silent | copy | 15.21 ± 1.12 | 13.54 ± 1.03 | 13.07 ± 1.03 | -2.14 ± 0.29 (0.002) | -0.47 ± 0.39 (0.049) |
| silent | scratch | 15.21 ± 1.12 | 13.54 ± 1.03 | 12.66 ± 0.70 | -2.55 ± 0.62 (0.002) | -0.87 ± 0.53 (0.0039) |

## Learning curves

Validation food (mean over runs) at quantiles of each run's curve (median generation in brackets). Full curves in `add_specialist_curves.csv`.

| condition | init | metric | q0 | q10 | q25 | q50 | q75 | q100 |
|---|---|---|---|---|---|---|---|---|
| learned | scratch | val_newcomer_food | 0.08 (g0) | 0.48 (g80) | 1.90 (g200) | 3.66 (g400) | 4.74 (g600) | 5.79 (g800) |
| learned | copy | val_newcomer_food | 9.87 (g0) | 10.00 (g10) | 9.67 (g20) | 9.93 (g50) | 9.97 (g80) | 9.62 (g100) |
| hand-coded | scratch | val_newcomer_food | 0.09 (g0) | 0.33 (g80) | 1.56 (g200) | 3.58 (g400) | 4.88 (g600) | 5.26 (g800) |
| hand-coded | copy | val_newcomer_food | 9.72 (g0) | 9.81 (g10) | 9.51 (g20) | 9.36 (g50) | 9.44 (g80) | 9.85 (g100) |
| joint | - | val_per_snake_food | 0.09 (g0) | 1.74 (g120) | 4.95 (g300) | 8.73 (g600) | 9.56 (g900) | 9.93 (g1200) |
| silent | scratch | val_newcomer_food | 0.10 (g0) | 0.25 (g80) | 0.45 (g200) | 0.77 (g400) | 1.00 (g600) | 1.29 (g800) |
| silent | copy | val_newcomer_food | 4.33 (g0) | 3.93 (g10) | 4.04 (g20) | 4.04 (g50) | 3.84 (g80) | 3.85 (g100) |
