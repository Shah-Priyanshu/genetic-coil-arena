# Squad: learned protocol vs hand-coded protocol at 3 bits (R6)

- Date: 2026-10-23
- Commit: b2e4442 (code; this report is committed on top of it)
- Compute: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, 28 logical cores, 20 worker processes, numpy/numba only (no GPU). Training wall time 522.3 min (142.19 CPU-hours), evaluation 1.4 min.
- Training: 10 independent runs per condition and radius; 2000 ES generations × population 128 × 8 episodes per member (≈2048k episodes per run).
- Held-out evaluation: run s plays episodes 3,500,000,000 + s × 10,000 + i, i < 200. Training runs are the independent units for every CI and test (n = number of runs), so the CIs include training-seed variance, not only episode noise.
- Squad: 21×21 walls, 3 snakes, 2 food per colour, health 60, T = 300; channel = 1 symbol, V = 8 (3 bits) per receiver, sent every step (k = 1, the hand-coded centre's interval), read one step later.

Reproduce with `python scripts/learned_protocol.py --scale gpu --radii 5 --runs 10 --generations 2000 --population 128 --episodes-per-member 8 --workers 20 --eval-episodes 200 --out docs/results/learned_protocol.md`.

## Design

- **Algorithm:** OpenAI-ES (`learning.EvolutionStrategy`: antithetic Gaussian perturbations σ = 0.05, centred-rank shaping, Adam lr = 0.03, weight decay 0.005). Centre and specialists are one parameter vector, so the joint system is optimised end-to-end from the team return through the discrete argmax channel, with no gradient through the message. All members of a generation play the same episodes and action noise. Fitness = team return (food + 0.01 per living snake per step). Same budget, sizes and hyperparameters for every condition; the three conditions of run s start from the same specialist weights.
- **Specialist (shared by the 3 snakes):** egocentric window planes (4·(2r+1)²), health, one-hot received code (8) → 32 tanh → 3 action logits, actions sampled from the softmax (15907 parameters at r = 5).
- **Learned centre (shared across receivers):** per receiver, its two own-colour food items as egocentric (ahead, right) / 21 + valid flag, sorted near to far, and its health (7 inputs) → 16 tanh → 8 logits; the code is the argmax (264 parameters). It gets food positions, not directions or distance classes: which partition of the plane to name is learned. Sorting by distance only removes a slot permutation.
- **Conditions:** (a) `silent+learned-spec`: the centre always sends code 0; (b) `hand-coded+learned-spec`: the fixed `HandCodedCentre` (`2·direction + far`); (c) `learned-centre+learned-spec`. Plus the scripted references `greedy+hand-coded` and the registered baselines (random actions, greedy + silent / random centre, no-comm greedy, full-view oracle).
- **Checkpoint:** the ES mean is scored every 10 generations on 32 fixed validation episodes (seeds 4,000,000,000 + run × 10,000 + i, disjoint from training and test); the best checkpoint is evaluated on the test seeds.
- **Metrics:** computed per training run and averaged (runs learn different codes, so messages are never pooled). T = the receiver's nearest own-colour food: direction relative to heading (4) × near/far (Manhattan > 5) + 'no food' (the hand-coded meaning). topsim = Spearman between Hamming distance of (direction, far) and of the code, on receivers with food. CIC from the specialists' exact softmax.

## Interpretation

- **r = 5:** team food (a) silent 14.05, (b) hand-coded 27.25, (c) learned 27.33; scripted greedy+hand-coded 26.51, no-comm 15.84, oracle 43.10. (c) − (a): +13.28 [+11.13, +15.42], higher, p(Holm) = 0.00091. (c) − (b): +0.07 [-1.67, +1.82], not distinguishable, p(Holm) = 0.850. (b) − (a): +13.20 [+10.95, +15.46], higher, p(Holm) = 0.00091. The learned centre uses 1.24 of 3 bits with MI(M;T) = 0.76 bits; CIC 0.458 bits (means over runs); interventions whose drop CI excludes 0: `zero`, `shuffle_batch`, `swap_receivers`, `random_matched`.

<!-- discussion -->

## Team food per episode (held-out)

Mean ± 95% t-CI over training runs (each run's score is its mean over the test episodes; scripted systems and baselines use the same per-run episode sets).

| condition | role | r = 5 |
|---|---|---|
| silent+learned-spec | system | 14.05 ± 1.93 |
| hand-coded+learned-spec | system | 27.25 ± 1.44 |
| learned-centre+learned-spec | system | 27.33 ± 1.20 |
| greedy+hand-coded | system | 26.51 ± 0.26 |
| random-actions | baseline | 0.33 ± 0.03 |
| greedy+silent-centre | baseline | 14.88 ± 0.30 |
| greedy+random-centre | baseline | 7.43 ± 0.11 |
| no-comm | baseline | 15.84 ± 0.57 |
| oracle | baseline | 43.10 ± 0.32 |

## Headline comparisons

(c) vs (a): is learned communication useful at all; (c) vs (b): learned vs hand-coded protocol for learned listeners; (b) vs (a): do learned listeners use the hand-coded channel. Unpaired over training runs.

| r | comparison | mean diff [95% Welch CI] | p (Mann-Whitney, Holm) | A12 |
|---|---|---|---|---|
| 5 | learned-centre+learned-spec vs silent+learned-spec | 13.28 [11.13, 15.42] | 0.00091 | 1.00 |
| 5 | learned-centre+learned-spec vs hand-coded+learned-spec | 0.07 [-1.67, 1.82] | 0.850 | 0.53 |
| 5 | hand-coded+learned-spec vs silent+learned-spec | 13.20 [10.95, 15.46] | 0.00091 | 1.00 |
| 5 | learned-centre+learned-spec vs greedy+hand-coded | 0.82 [-0.40, 2.03] | 0.558 | 0.68 |
| 5 | hand-coded+learned-spec vs greedy+hand-coded | 0.75 [-0.71, 2.20] | 0.769 | 0.62 |

## Message ablations (learned systems)

Drop = intact − ablated team food, paired by training run; p = Wilcoxon signed-rank, Holm across the four interventions per system.

| r | system | intervention | intact | ablated | drop [95% CI] | p (Holm) | A12 | listening |
|---|---|---|---|---|---|---|---|---|
| 5 | silent+learned-spec | zero | 14.05 | 14.05 | 0.00 [0.00, 0.00] | 1.000 | 0.50 | no |
| 5 | silent+learned-spec | shuffle_batch | 14.05 | 14.05 | 0.00 [0.00, 0.00] | 1.000 | 0.50 | no |
| 5 | silent+learned-spec | swap_receivers | 14.05 | 14.05 | 0.00 [0.00, 0.00] | 1.000 | 0.50 | no |
| 5 | silent+learned-spec | random_matched | 14.05 | 14.05 | 0.00 [0.00, 0.00] | 1.000 | 0.50 | no |
| 5 | hand-coded+learned-spec | zero | 27.25 | 3.55 | 23.70 [22.71, 24.70] | 0.0078 | 1.00 | yes |
| 5 | hand-coded+learned-spec | shuffle_batch | 27.25 | 2.33 | 24.92 [23.54, 26.30] | 0.0078 | 1.00 | yes |
| 5 | hand-coded+learned-spec | swap_receivers | 27.25 | 8.15 | 19.10 [18.09, 20.11] | 0.0078 | 1.00 | yes |
| 5 | hand-coded+learned-spec | random_matched | 27.25 | 2.42 | 24.83 [23.41, 26.25] | 0.0078 | 1.00 | yes |
| 5 | learned-centre+learned-spec | zero | 27.33 | 2.42 | 24.91 [23.23, 26.58] | 0.0078 | 1.00 | yes |
| 5 | learned-centre+learned-spec | shuffle_batch | 27.33 | 1.98 | 25.35 [23.75, 26.95] | 0.0078 | 1.00 | yes |
| 5 | learned-centre+learned-spec | swap_receivers | 27.33 | 7.08 | 20.25 [18.79, 21.72] | 0.0078 | 1.00 | yes |
| 5 | learned-centre+learned-spec | random_matched | 27.33 | 1.96 | 25.37 [23.83, 26.90] | 0.0078 | 1.00 | yes |

## What the centre says and how much it moves actions

Mean ± 95% CI over training runs; MI null q95 from 100 target shuffles (mean over runs).

| r | squad | H(M) bits | used / 3 bits | codes used | H(T) bits | MI(M;T) bits | MI null q95 | MI(M;direction) | MI(M;far) | topsim | CIC bits |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | silent+learned-spec | 0.000 ± 0.000 | 0% | 1.0 ± 0.0 | 2.515 ± 0.040 | 0.000 ± 0.000 | 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | n/a | n/a |
| 5 | hand-coded+learned-spec | 2.696 ± 0.043 | 90% | 8.0 ± 0.0 | 2.696 ± 0.043 | 2.696 ± 0.043 | 0.000 | 1.764 ± 0.033 | 0.942 ± 0.010 | 0.730 ± 0.015 | 0.464 ± 0.049 |
| 5 | learned-centre+learned-spec | 1.235 ± 0.129 | 41% | 3.7 ± 0.6 | 2.656 ± 0.020 | 0.761 ± 0.080 | 0.000 | 0.686 ± 0.089 | 0.031 ± 0.024 | 0.329 ± 0.039 | 0.458 ± 0.042 |
| 5 | greedy+hand-coded | 2.606 ± 0.004 | 87% | 8.0 ± 0.0 | 2.606 ± 0.004 | 2.606 ± 0.004 | 0.000 | 1.827 ± 0.002 | 0.830 ± 0.004 | 0.746 ± 0.003 | 0.646 ± 0.009 |

| r | squad | JS(shuffle_batch) bits | action changed (shuffle_batch) | JS(random_matched) bits | action changed (random_matched) |
|---|---|---|---|---|---|
| 5 | hand-coded+learned-spec | 0.183 ± 0.030 | 0.223 ± 0.039 | 0.185 ± 0.029 | 0.226 ± 0.038 |
| 5 | learned-centre+learned-spec | 0.217 ± 0.020 | 0.257 ± 0.022 | 0.216 ± 0.020 | 0.256 ± 0.021 |
| 5 | greedy+hand-coded | 0.304 ± 0.006 | 0.311 ± 0.006 | 0.302 ± 0.007 | 0.309 ± 0.008 |

### Learned centre, run by run

(a), (b), (c) = silent, hand-coded and learned centre with learned specialists trained from the same initial specialist weights.

| r | run | team food (c) | (b) | (a) | H(M) bits | codes used | MI(M;T) | MI(M;direction) | MI(M;far) | CIC bits | drop when zeroed | drop when shuffled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 0 | 26.14 | 27.67 | 8.65 | 1.38 | 3 | 0.85 | 0.78 | 0.06 | 0.424 | 23.14 | 24.42 |
| 5 | 1 | 28.82 | 27.36 | 16.95 | 1.31 | 3 | 0.64 | 0.51 | 0.03 | 0.476 | 26.95 | 27.57 |
| 5 | 2 | 27.18 | 25.86 | 15.79 | 1.00 | 5 | 0.74 | 0.71 | 0.00 | 0.462 | 25.08 | 24.81 |
| 5 | 3 | 30.18 | 29.98 | 11.55 | 1.35 | 3 | 0.73 | 0.61 | 0.02 | 0.447 | 29.36 | 28.74 |
| 5 | 4 | 28.59 | 27.93 | 17.43 | 1.55 | 4 | 0.95 | 0.84 | 0.11 | 0.568 | 26.34 | 26.68 |
| 5 | 5 | 26.77 | 25.88 | 13.87 | 1.33 | 4 | 0.84 | 0.77 | 0.04 | 0.405 | 24.72 | 24.88 |
| 5 | 6 | 28.27 | 27.24 | 15.77 | 1.12 | 3 | 0.56 | 0.46 | 0.01 | 0.356 | 25.43 | 27.12 |
| 5 | 7 | 26.85 | 25.11 | 13.60 | 1.01 | 5 | 0.72 | 0.68 | 0.00 | 0.478 | 23.92 | 24.55 |
| 5 | 8 | 26.14 | 24.64 | 14.79 | 1.24 | 3 | 0.78 | 0.71 | 0.04 | 0.522 | 23.18 | 23.75 |
| 5 | 9 | 24.32 | 30.89 | 12.11 | 1.06 | 4 | 0.81 | 0.80 | 0.01 | 0.444 | 20.95 | 20.98 |

### Learned codebooks

r = 5, training run 0 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 6310 | 8845 | 68 | 241 | 0 | 0 | 597 | 1601 | 0 |
| 3 | 17 | 323 | 60 | 3616 | 0 | 0 | 0 | 0 | 0 |
| 4 | 643 | 0 | 3858 | 5805 | 1289 | 3579 | 593 | 636 | 0 |

r = 5, training run 1 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1355 | 14 | 175 | 6 | 20 | 1 | 493 | 1000 | 0 |
| 5 | 5602 | 9612 | 301 | 3959 | 0 | 0 | 36 | 428 | 0 |
| 6 | 1282 | 0 | 3455 | 5286 | 1544 | 3742 | 598 | 635 | 0 |

r = 5, training run 2 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 5858 | 9839 | 49 | 432 | 8 | 0 | 1033 | 1720 | 0 |
| 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 2 | 0 | 1 | 0 | 0 | 0 | 0 |
| 4 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 0 |
| 6 | 1086 | 154 | 5082 | 8875 | 1486 | 3656 | 212 | 230 | 0 |

r = 5, training run 3 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1319 | 0 | 3508 | 5197 | 1305 | 3469 | 329 | 125 | 0 |
| 1 | 1016 | 0 | 39 | 0 | 22 | 97 | 942 | 1792 | 0 |
| 2 | 6043 | 10140 | 277 | 3907 | 0 | 0 | 28 | 635 | 0 |

r = 5, training run 4 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 6790 | 9108 | 57 | 231 | 1 | 0 | 990 | 1550 | 0 |
| 3 | 518 | 27 | 4123 | 4694 | 1428 | 3096 | 120 | 181 | 0 |
| 5 | 1 | 494 | 103 | 6644 | 1 | 72 | 0 | 0 | 0 |
| 7 | 86 | 40 | 59 | 140 | 0 | 0 | 8 | 0 | 0 |

r = 5, training run 5 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 8 | 0 | 2 | 1 | 0 | 0 | 1 | 0 | 0 |
| 3 | 745 | 17 | 409 | 450 | 1335 | 4168 | 4333 | 7046 | 0 |
| 4 | 6375 | 9805 | 635 | 1374 | 0 | 0 | 1 | 576 | 0 |
| 6 | 11 | 116 | 0 | 0 | 2 | 1 | 44 | 3186 | 0 |

r = 5, training run 6 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 2023 | 0 | 3229 | 6077 | 1320 | 4321 | 1011 | 1399 | 0 |
| 4 | 38 | 0 | 0 | 0 | 0 | 0 | 108 | 647 | 0 |
| 7 | 5830 | 10342 | 306 | 4191 | 0 | 0 | 50 | 682 | 0 |

r = 5, training run 7 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 0 | 0 | 3 | 2 | 12 | 5 | 0 | 0 |
| 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | 1729 | 472 | 323 | 114 | 1375 | 3283 | 4285 | 8648 | 0 |
| 6 | 4902 | 9533 | 1348 | 2136 | 3 | 1 | 9 | 56 | 0 |
| 7 | 1 | 14 | 0 | 0 | 0 | 0 | 0 | 6 | 0 |

r = 5, training run 8 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1117 | 109 | 4342 | 7441 | 1564 | 3313 | 489 | 239 | 0 |
| 5 | 5715 | 9261 | 26 | 518 | 0 | 0 | 828 | 1505 | 0 |
| 6 | 0 | 34 | 1 | 1990 | 0 | 0 | 0 | 0 | 0 |

r = 5, training run 9 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 6176 | 9479 | 842 | 1447 | 8 | 0 | 100 | 549 | 0 |
| 4 | 364 | 28 | 102 | 152 | 1510 | 3928 | 3922 | 8601 | 0 |
| 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 176 | 0 |
| 7 | 5 | 2 | 0 | 0 | 3 | 0 | 12 | 81 | 0 |

## Learning curves

Full curves (every 10 generations, every run) in `learned_protocol_curves.csv`.

r = 5 (validation team food of the ES mean, mean over runs):

| condition | gen 0 | gen 200 | gen 500 | gen 1000 | gen 1500 | gen 2000 | best ckpt gen (mean) |
|---|---|---|---|---|---|---|---|
| silent+learned-spec | 0.37 | 2.67 | 5.47 | 10.12 | 12.38 | 13.75 | 1776 |
| hand-coded+learned-spec | 0.37 | 12.37 | 22.43 | 25.82 | 26.67 | 28.11 | 1611 |
| learned-centre+learned-spec | 0.37 | 7.76 | 18.85 | 24.30 | 27.12 | 26.54 | 1743 |

## Harness output

`evaluation.format_table` per radius.

### r = 5

```
Task: squad-learned-protocol-r5  (seeds: 10)
Baselines: random-actions, greedy+silent-centre, greedy+random-centre, no-comm, oracle

condition                    role      n   mean ± 95% CI   IQM [95% CI]           
---------------------------  --------  --  --------------  -----------------------
silent+learned-spec          system    10  14.050 ± 1.934  14.321 [12.261, 15.972]
hand-coded+learned-spec      system    10  27.254 ± 1.442  26.988 [25.828, 28.570]
learned-centre+learned-spec  system    10  27.325 ± 1.199  27.299 [26.282, 28.428]
greedy+hand-coded            system    10  26.509 ± 0.263  26.536 [26.307, 26.737]
random-actions               baseline  10  0.325 ± 0.027   0.321 [0.302, 0.347]   
greedy+silent-centre         baseline  10  14.875 ± 0.295  14.909 [14.575, 15.193]
greedy+random-centre         baseline  10  7.426 ± 0.113   7.422 [7.313, 7.526]   
no-comm                      baseline  10  15.843 ± 0.572  15.948 [15.341, 16.399]
oracle                       baseline  10  43.100 ± 0.315  43.145 [42.829, 43.393]

system                       intervention    ablated mean  drop [95% CI]            p (Holm)  A12    listening
---------------------------  --------------  ------------  -----------------------  --------  -----  ---------
silent+learned-spec          zero            14.050        0.000 [0.000, 0.000]     1.000     0.500  no       
silent+learned-spec          shuffle_batch   14.050        0.000 [0.000, 0.000]     1.000     0.500  no       
silent+learned-spec          swap_receivers  14.050        0.000 [0.000, 0.000]     1.000     0.500  no       
silent+learned-spec          random_matched  14.050        0.000 [0.000, 0.000]     1.000     0.500  no       
hand-coded+learned-spec      zero            3.551         23.703 [22.709, 24.696]  0.008     1.000  yes      
hand-coded+learned-spec      shuffle_batch   2.333         24.920 [23.539, 26.302]  0.008     1.000  yes      
hand-coded+learned-spec      swap_receivers  8.152         19.102 [18.093, 20.111]  0.008     1.000  yes      
hand-coded+learned-spec      random_matched  2.421         24.834 [23.415, 26.252]  0.008     1.000  yes      
learned-centre+learned-spec  zero            2.419         24.907 [23.235, 26.579]  0.008     1.000  yes      
learned-centre+learned-spec  shuffle_batch   1.976         25.349 [23.749, 26.950]  0.008     1.000  yes      
learned-centre+learned-spec  swap_receivers  7.075         20.250 [18.786, 21.715]  0.008     1.000  yes      
learned-centre+learned-spec  random_matched  1.961         25.365 [23.834, 26.896]  0.008     1.000  yes      

comparison                                           mean diff  p      p (Holm)  A12  
---------------------------------------------------  ---------  -----  --------  -----
silent+learned-spec vs random-actions                13.725     0.000  0.004     1.000
silent+learned-spec vs greedy+silent-centre          -0.825     0.678  0.678     0.440
silent+learned-spec vs greedy+random-centre          6.624      0.000  0.004     1.000
silent+learned-spec vs no-comm                       -1.793     0.162  0.324     0.310
silent+learned-spec vs oracle                        -29.050    0.000  0.004     0.000
hand-coded+learned-spec vs random-actions            26.929     0.000  0.004     1.000
hand-coded+learned-spec vs greedy+silent-centre      12.379     0.000  0.004     1.000
hand-coded+learned-spec vs greedy+random-centre      19.828     0.000  0.004     1.000
hand-coded+learned-spec vs no-comm                   11.411     0.000  0.004     1.000
hand-coded+learned-spec vs oracle                    -15.846    0.000  0.004     0.000
learned-centre+learned-spec vs random-actions        27.000     0.000  0.004     1.000
learned-centre+learned-spec vs greedy+silent-centre  12.450     0.000  0.004     1.000
learned-centre+learned-spec vs greedy+random-centre  19.899     0.000  0.004     1.000
learned-centre+learned-spec vs no-comm               11.482     0.000  0.004     1.000
learned-centre+learned-spec vs oracle                -15.775    0.000  0.004     0.000
greedy+hand-coded vs random-actions                  26.184     0.000  0.004     1.000
greedy+hand-coded vs greedy+silent-centre            11.633     0.000  0.004     1.000
greedy+hand-coded vs greedy+random-centre            19.083     0.000  0.004     1.000
greedy+hand-coded vs no-comm                         10.666     0.000  0.004     1.000
greedy+hand-coded vs oracle                          -16.591    0.000  0.004     0.000

system                       changed_random_matched  changed_random_matched_ci  changed_shuffle_batch  changed_shuffle_batch_ci  cic_bits  cic_bits_ci  distinct_codes  distinct_codes_ci  js_random_matched_bits  js_random_matched_bits_ci  js_shuffle_batch_bits  js_shuffle_batch_bits_ci  message_entropy_bits  message_entropy_bits_ci  mi_bits  mi_bits_ci  mi_direction_bits  mi_direction_bits_ci  mi_far_bits  mi_far_bits_ci  mi_null_q95_bits  mi_null_q95_bits_ci  mi_p_value  mi_p_value_ci  target_entropy_bits  target_entropy_bits_ci  topsim  topsim_ci  utilization  utilization_ci
---------------------------  ----------------------  -------------------------  ---------------------  ------------------------  --------  -----------  --------------  -----------------  ----------------------  -------------------------  ---------------------  ------------------------  --------------------  -----------------------  -------  ----------  -----------------  --------------------  -----------  --------------  ----------------  -------------------  ----------  -------------  -------------------  ----------------------  ------  ---------  -----------  --------------
silent+learned-spec          -                       -                          -                      -                         -         -            1.000           0.000              -                       -                          -                      -                         0.000                 0.000                    0.000    0.000       0.000              0.000                 0.000        0.000           0.000             0.000                1.000       0.000          2.515                0.040                   nan     nan        0.000        0.000         
hand-coded+learned-spec      0.226                   0.038                      0.223                  0.039                     0.464     0.049        8.000           0.000              0.185                   0.029                      0.183                  0.030                     2.696                 0.043                    2.696    0.043       1.764              0.033                 0.942        0.010           0.000             0.000                0.010       0.000          2.696                0.043                   0.730   0.015      0.899        0.014         
learned-centre+learned-spec  0.256                   0.021                      0.257                  0.022                     0.458     0.042        3.700           0.589              0.216                   0.020                      0.217                  0.020                     1.235                 0.129                    0.761    0.080       0.686              0.089                 0.031        0.024           0.000             0.000                0.010       0.000          2.656                0.020                   0.329   0.039      0.412        0.043         
greedy+hand-coded            0.309                   0.008                      0.311                  0.006                     0.646     0.009        8.000           0.000              0.302                   0.007                      0.304                  0.006                     2.606                 0.004                    2.606    0.004       1.827              0.002                 0.830        0.004           0.000             0.000                0.010       0.000          2.606                0.004                   0.746   0.003      0.869        0.001         
```
