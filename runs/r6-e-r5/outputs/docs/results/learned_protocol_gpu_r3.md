# Squad: learned protocol vs hand-coded protocol at 3 bits (R6)

- Date: 2026-10-23
- Commit: dc65462 (code; this report is committed on top of it)
- Compute: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, 28 logical cores, 20 worker processes, numpy/numba only (no GPU). Training wall time 218.1 min (59.96 CPU-hours), evaluation 0.7 min.
- Training: 10 independent runs per condition and radius; 2000 ES generations × population 128 × 8 episodes per member (≈2048k episodes per run).
- Held-out evaluation: run s plays episodes 3,500,000,000 + s × 10,000 + i, i < 200. Training runs are the independent units for every CI and test (n = number of runs), so the CIs include training-seed variance, not only episode noise.
- Squad: 21×21 walls, 3 snakes, 2 food per colour, health 60, T = 300; channel = 1 symbol, V = 8 (3 bits) per receiver, sent every step (k = 1, the hand-coded centre's interval), read one step later.

Reproduce with `python scripts/learned_protocol.py --scale gpu --radii 3 --runs 10 --generations 2000 --population 128 --episodes-per-member 8 --workers 20 --eval-episodes 200 --out docs/results/learned_protocol.md`.

## Design

- **Algorithm:** OpenAI-ES (`learning.EvolutionStrategy`: antithetic Gaussian perturbations σ = 0.05, centred-rank shaping, Adam lr = 0.03, weight decay 0.005). Centre and specialists are one parameter vector, so the joint system is optimised end-to-end from the team return through the discrete argmax channel, with no gradient through the message. All members of a generation play the same episodes and action noise. Fitness = team return (food + 0.01 per living snake per step). Same budget, sizes and hyperparameters for every condition; the three conditions of run s start from the same specialist weights.
- **Specialist (shared by the 3 snakes):** egocentric window planes (4·(2r+1)²), health, one-hot received code (8) → 16 tanh → 3 action logits, actions sampled from the softmax (3347 parameters at r = 3).
- **Learned centre (shared across receivers):** per receiver, its two own-colour food items as egocentric (ahead, right) / 21 + valid flag, sorted near to far, and its health (7 inputs) → 16 tanh → 8 logits; the code is the argmax (264 parameters). It gets food positions, not directions or distance classes: which partition of the plane to name is learned. Sorting by distance only removes a slot permutation.
- **Conditions:** (a) `silent+learned-spec`: the centre always sends code 0; (b) `hand-coded+learned-spec`: the fixed `HandCodedCentre` (`2·direction + far`); (c) `learned-centre+learned-spec`. Plus the scripted references `greedy+hand-coded` and the registered baselines (random actions, greedy + silent / random centre, no-comm greedy, full-view oracle).
- **Checkpoint:** the ES mean is scored every 10 generations on 32 fixed validation episodes (seeds 4,000,000,000 + run × 10,000 + i, disjoint from training and test); the best checkpoint is evaluated on the test seeds.
- **Metrics:** computed per training run and averaged (runs learn different codes, so messages are never pooled). T = the receiver's nearest own-colour food: direction relative to heading (4) × near/far (Manhattan > 5) + 'no food' (the hand-coded meaning). topsim = Spearman between Hamming distance of (direction, far) and of the code, on receivers with food. CIC from the specialists' exact softmax.

## Interpretation

- **r = 3:** team food (a) silent 15.21, (b) hand-coded 34.89, (c) learned 35.43; scripted greedy+hand-coded 18.70, no-comm 8.10, oracle 43.10. (c) − (a): +20.22 [+18.40, +22.04], higher, p(Holm) = 0.00091. (c) − (b): +0.54 [-2.14, +3.22], not distinguishable, p(Holm) = 0.970. (b) − (a): +19.67 [+17.17, +22.18], higher, p(Holm) = 0.00091. The learned centre uses 1.19 of 3 bits with MI(M;T) = 0.73 bits; CIC 0.471 bits (means over runs); interventions whose drop CI excludes 0: `zero`, `shuffle_batch`, `swap_receivers`, `random_matched`.

<!-- discussion -->

## Team food per episode (held-out)

Mean ± 95% t-CI over training runs (each run's score is its mean over the test episodes; scripted systems and baselines use the same per-run episode sets).

| condition | role | r = 3 |
|---|---|---|
| silent+learned-spec | system | 15.21 ± 1.12 |
| hand-coded+learned-spec | system | 34.89 ± 2.37 |
| learned-centre+learned-spec | system | 35.43 ± 1.59 |
| greedy+hand-coded | system | 18.70 ± 0.33 |
| random-actions | baseline | 0.33 ± 0.03 |
| greedy+silent-centre | baseline | 6.70 ± 0.22 |
| greedy+random-centre | baseline | 3.76 ± 0.16 |
| no-comm | baseline | 8.10 ± 0.27 |
| oracle | baseline | 43.10 ± 0.32 |

## Headline comparisons

(c) vs (a): is learned communication useful at all; (c) vs (b): learned vs hand-coded protocol for learned listeners; (b) vs (a): do learned listeners use the hand-coded channel. Unpaired over training runs.

| r | comparison | mean diff [95% Welch CI] | p (Mann-Whitney, Holm) | A12 |
|---|---|---|---|---|
| 3 | learned-centre+learned-spec vs silent+learned-spec | 20.22 [18.40, 22.04] | 0.00091 | 1.00 |
| 3 | learned-centre+learned-spec vs hand-coded+learned-spec | 0.54 [-2.14, 3.22] | 0.970 | 0.49 |
| 3 | hand-coded+learned-spec vs silent+learned-spec | 19.67 [17.17, 22.18] | 0.00091 | 1.00 |
| 3 | learned-centre+learned-spec vs greedy+hand-coded | 16.73 [15.13, 18.33] | 0.00091 | 1.00 |
| 3 | hand-coded+learned-spec vs greedy+hand-coded | 16.19 [13.80, 18.57] | 0.00091 | 1.00 |

## Message ablations (learned systems)

Drop = intact − ablated team food, paired by training run; p = Wilcoxon signed-rank, Holm across the four interventions per system.

| r | system | intervention | intact | ablated | drop [95% CI] | p (Holm) | A12 | listening |
|---|---|---|---|---|---|---|---|---|
| 3 | silent+learned-spec | zero | 15.21 | 15.21 | 0.00 [0.00, 0.00] | 1.000 | 0.50 | no |
| 3 | silent+learned-spec | shuffle_batch | 15.21 | 15.21 | 0.00 [0.00, 0.00] | 1.000 | 0.50 | no |
| 3 | silent+learned-spec | swap_receivers | 15.21 | 15.21 | 0.00 [0.00, 0.00] | 1.000 | 0.50 | no |
| 3 | silent+learned-spec | random_matched | 15.21 | 15.21 | 0.00 [0.00, 0.00] | 1.000 | 0.50 | no |
| 3 | hand-coded+learned-spec | zero | 34.89 | 4.23 | 30.66 [28.57, 32.75] | 0.0078 | 1.00 | yes |
| 3 | hand-coded+learned-spec | shuffle_batch | 34.89 | 2.42 | 32.47 [30.24, 34.70] | 0.0078 | 1.00 | yes |
| 3 | hand-coded+learned-spec | swap_receivers | 34.89 | 9.62 | 25.26 [23.73, 26.80] | 0.0078 | 1.00 | yes |
| 3 | hand-coded+learned-spec | random_matched | 34.89 | 2.43 | 32.46 [30.24, 34.68] | 0.0078 | 1.00 | yes |
| 3 | learned-centre+learned-spec | zero | 35.43 | 3.53 | 31.91 [29.56, 34.25] | 0.0078 | 1.00 | yes |
| 3 | learned-centre+learned-spec | shuffle_batch | 35.43 | 2.60 | 32.83 [30.99, 34.67] | 0.0078 | 1.00 | yes |
| 3 | learned-centre+learned-spec | swap_receivers | 35.43 | 9.28 | 26.16 [24.71, 27.61] | 0.0078 | 1.00 | yes |
| 3 | learned-centre+learned-spec | random_matched | 35.43 | 2.82 | 32.62 [30.80, 34.44] | 0.0078 | 1.00 | yes |

## What the centre says and how much it moves actions

Mean ± 95% CI over training runs; MI null q95 from 100 target shuffles (mean over runs).

| r | squad | H(M) bits | used / 3 bits | codes used | H(T) bits | MI(M;T) bits | MI null q95 | MI(M;direction) | MI(M;far) | topsim | CIC bits |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | silent+learned-spec | 0.000 ± 0.000 | 0% | 1.0 ± 0.0 | 2.441 ± 0.058 | 0.000 ± 0.000 | 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | n/a | n/a |
| 3 | hand-coded+learned-spec | 2.654 ± 0.028 | 88% | 8.0 ± 0.0 | 2.654 ± 0.028 | 2.654 ± 0.028 | 0.000 | 1.731 ± 0.018 | 0.932 ± 0.010 | 0.740 ± 0.007 | 0.489 ± 0.048 |
| 3 | learned-centre+learned-spec | 1.187 ± 0.180 | 40% | 4.1 ± 1.2 | 2.596 ± 0.036 | 0.727 ± 0.029 | 0.000 | 0.667 ± 0.046 | 0.021 ± 0.016 | 0.349 ± 0.041 | 0.471 ± 0.026 |
| 3 | greedy+hand-coded | 2.631 ± 0.004 | 88% | 8.0 ± 0.0 | 2.631 ± 0.004 | 2.631 ± 0.004 | 0.000 | 1.907 ± 0.003 | 0.768 ± 0.004 | 0.746 ± 0.004 | 0.854 ± 0.006 |

| r | squad | JS(shuffle_batch) bits | action changed (shuffle_batch) | JS(random_matched) bits | action changed (random_matched) |
|---|---|---|---|---|---|
| 3 | hand-coded+learned-spec | 0.193 ± 0.025 | 0.242 ± 0.031 | 0.195 ± 0.026 | 0.242 ± 0.032 |
| 3 | learned-centre+learned-spec | 0.221 ± 0.012 | 0.267 ± 0.016 | 0.222 ± 0.011 | 0.269 ± 0.014 |
| 3 | greedy+hand-coded | 0.401 ± 0.008 | 0.413 ± 0.010 | 0.400 ± 0.008 | 0.411 ± 0.008 |

### Learned centre, run by run

(a), (b), (c) = silent, hand-coded and learned centre with learned specialists trained from the same initial specialist weights.

| r | run | team food (c) | (b) | (a) | H(M) bits | codes used | MI(M;T) | MI(M;direction) | MI(M;far) | CIC bits | drop when zeroed | drop when shuffled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0 | 34.99 | 38.22 | 14.72 | 1.01 | 3 | 0.74 | 0.70 | 0.00 | 0.451 | 30.39 | 32.55 |
| 3 | 1 | 36.80 | 37.31 | 15.02 | 1.33 | 3 | 0.69 | 0.56 | 0.04 | 0.533 | 32.75 | 35.05 |
| 3 | 2 | 34.16 | 34.97 | 15.19 | 1.05 | 7 | 0.67 | 0.64 | 0.01 | 0.444 | 32.37 | 31.94 |
| 3 | 3 | 36.06 | 36.09 | 16.66 | 1.01 | 3 | 0.76 | 0.73 | 0.00 | 0.418 | 30.62 | 33.31 |
| 3 | 4 | 34.24 | 35.20 | 14.68 | 1.00 | 3 | 0.77 | 0.75 | 0.01 | 0.420 | 31.13 | 31.42 |
| 3 | 5 | 38.47 | 35.12 | 17.67 | 1.43 | 4 | 0.76 | 0.60 | 0.06 | 0.501 | 33.35 | 35.53 |
| 3 | 6 | 30.39 | 29.64 | 12.26 | 0.97 | 2 | 0.71 | 0.69 | 0.02 | 0.486 | 24.30 | 26.62 |
| 3 | 7 | 36.66 | 36.69 | 15.48 | 1.39 | 7 | 0.74 | 0.62 | 0.03 | 0.483 | 35.56 | 34.72 |
| 3 | 8 | 37.06 | 28.30 | 16.74 | 1.69 | 5 | 0.77 | 0.72 | 0.05 | 0.479 | 36.06 | 34.21 |
| 3 | 9 | 35.48 | 37.35 | 13.72 | 0.99 | 4 | 0.67 | 0.64 | 0.00 | 0.494 | 32.51 | 32.97 |

### Learned codebooks

r = 3, training run 0 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 1407 | 204 | 379 | 341 | 1129 | 4074 | 5144 | 9619 | 0 |
| 5 | 14 | 20 | 2 | 0 | 0 | 0 | 1 | 26 | 0 |
| 6 | 6698 | 12427 | 1058 | 2140 | 0 | 0 | 17 | 327 | 0 |

r = 3, training run 1 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 2093 | 4 | 630 | 1264 | 0 | 0 | 24 | 0 | 0 |
| 3 | 6603 | 11826 | 1 | 420 | 0 | 0 | 347 | 4582 | 0 |
| 7 | 1052 | 0 | 344 | 601 | 1063 | 3834 | 3558 | 5561 | 0 |

r = 3, training run 2 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 7923 | 12396 | 270 | 2121 | 1 | 0 | 762 | 1643 | 0 |
| 2 | 494 | 1 | 4267 | 8891 | 1127 | 4606 | 268 | 698 | 0 |
| 3 | 122 | 0 | 14 | 12 | 1 | 4 | 26 | 74 | 0 |
| 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
| 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 28 | 0 |
| 6 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 |
| 7 | 13 | 0 | 7 | 2 | 2 | 0 | 8 | 1 | 0 |

r = 3, training run 3 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 1430 | 303 | 135 | 181 | 1191 | 4189 | 5380 | 10925 | 0 |
| 6 | 6820 | 12980 | 1089 | 1968 | 0 | 0 | 1 | 328 | 0 |
| 7 | 11 | 2 | 0 | 0 | 3 | 0 | 3 | 2 | 0 |

r = 3, training run 4 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0 | 0 | 2 | 3 | 1 | 3 | 0 |
| 2 | 7653 | 12089 | 1316 | 2011 | 15 | 8 | 55 | 778 | 0 |
| 4 | 546 | 33 | 250 | 228 | 944 | 4306 | 3993 | 10148 | 0 |

r = 3, training run 5 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 3210 | 151 | 947 | 1856 | 34 | 50 | 15 | 0 | 0 |
| 1 | 1147 | 0 | 92 | 155 | 756 | 3867 | 3448 | 6010 | 0 |
| 4 | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 | 5149 | 12298 | 10 | 212 | 0 | 0 | 323 | 4708 | 0 |

r = 3, training run 6 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 7658 | 10820 | 1733 | 1165 | 139 | 71 | 1703 | 2546 | 0 |
| 2 | 7 | 0 | 3348 | 8767 | 1477 | 3852 | 0 | 17 | 0 |

r = 3, training run 7 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 0 | 3 | 0 | 0 | 0 | 2 | 1 | 0 |
| 1 | 6100 | 12304 | 465 | 5110 | 0 | 0 | 0 | 63 | 0 |
| 2 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | 2208 | 255 | 68 | 0 | 35 | 47 | 870 | 2070 | 0 |
| 4 | 1168 | 0 | 3642 | 5673 | 1180 | 4009 | 239 | 99 | 0 |
| 5 | 1 | 0 | 5 | 0 | 0 | 0 | 1 | 0 | 0 |
| 7 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |

r = 3, training run 8 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 216 | 0 | 2706 | 5173 | 838 | 3152 | 38 | 9 | 0 |
| 1 | 0 | 21 | 22 | 3226 | 4 | 0 | 0 | 0 | 0 |
| 3 | 1120 | 2028 | 1 | 0 | 99 | 155 | 877 | 2046 | 0 |
| 4 | 72 | 27 | 3 | 1 | 7 | 3 | 9 | 3 | 0 |
| 6 | 7734 | 9733 | 2010 | 4422 | 6 | 0 | 5 | 0 | 0 |

r = 3, training run 9 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 7631 | 12457 | 362 | 1758 | 10 | 4 | 832 | 1687 | 0 |
| 3 | 738 | 12 | 4582 | 9180 | 1013 | 4076 | 179 | 271 | 0 |
| 5 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| 6 | 0 | 0 | 0 | 7 | 0 | 0 | 0 | 0 | 0 |

## Learning curves

Full curves (every 10 generations, every run) in `learned_protocol_curves.csv`.

r = 3 (validation team food of the ES mean, mean over runs):

| condition | gen 0 | gen 200 | gen 500 | gen 1000 | gen 1500 | gen 2000 | best ckpt gen (mean) |
|---|---|---|---|---|---|---|---|
| silent+learned-spec | 0.35 | 3.58 | 9.55 | 12.93 | 14.31 | 15.64 | 1689 |
| hand-coded+learned-spec | 0.28 | 22.33 | 30.88 | 33.41 | 33.80 | 34.56 | 1698 |
| learned-centre+learned-spec | 0.30 | 13.66 | 29.89 | 34.61 | 34.79 | 36.13 | 1771 |

## Harness output

`evaluation.format_table` per radius.

### r = 3

```
Task: squad-learned-protocol-r3  (seeds: 10)
Baselines: random-actions, greedy+silent-centre, greedy+random-centre, no-comm, oracle

condition                    role      n   mean ± 95% CI   IQM [95% CI]           
---------------------------  --------  --  --------------  -----------------------
silent+learned-spec          system    10  15.215 ± 1.120  15.292 [14.243, 16.386]
hand-coded+learned-spec      system    10  34.889 ± 2.372  35.897 [32.754, 37.018]
learned-centre+learned-spec  system    10  35.432 ± 1.588  35.707 [34.200, 36.887]
greedy+hand-coded            system    10  18.703 ± 0.325  18.724 [18.336, 19.061]
random-actions               baseline  10  0.325 ± 0.027   0.321 [0.302, 0.347]   
greedy+silent-centre         baseline  10  6.696 ± 0.217   6.701 [6.495, 6.903]   
greedy+random-centre         baseline  10  3.761 ± 0.162   3.732 [3.593, 3.926]   
no-comm                      baseline  10  8.098 ± 0.272   8.102 [7.856, 8.354]   
oracle                       baseline  10  43.100 ± 0.315  43.145 [42.829, 43.393]

system                       intervention    ablated mean  drop [95% CI]            p (Holm)  A12    listening
---------------------------  --------------  ------------  -----------------------  --------  -----  ---------
silent+learned-spec          zero            15.215        0.000 [0.000, 0.000]     1.000     0.500  no       
silent+learned-spec          shuffle_batch   15.215        0.000 [0.000, 0.000]     1.000     0.500  no       
silent+learned-spec          swap_receivers  15.215        0.000 [0.000, 0.000]     1.000     0.500  no       
silent+learned-spec          random_matched  15.215        0.000 [0.000, 0.000]     1.000     0.500  no       
hand-coded+learned-spec      zero            4.231         30.658 [28.565, 32.751]  0.008     1.000  yes      
hand-coded+learned-spec      shuffle_batch   2.422         32.466 [30.237, 34.696]  0.008     1.000  yes      
hand-coded+learned-spec      swap_receivers  9.625         25.264 [23.728, 26.800]  0.008     1.000  yes      
hand-coded+learned-spec      random_matched  2.427         32.461 [30.244, 34.678]  0.008     1.000  yes      
learned-centre+learned-spec  zero            3.526         31.906 [29.565, 34.247]  0.008     1.000  yes      
learned-centre+learned-spec  shuffle_batch   2.599         32.832 [30.993, 34.672]  0.008     1.000  yes      
learned-centre+learned-spec  swap_receivers  9.277         26.155 [24.706, 27.605]  0.008     1.000  yes      
learned-centre+learned-spec  random_matched  2.817         32.615 [30.795, 34.436]  0.008     1.000  yes      

comparison                                           mean diff  p      p (Holm)  A12  
---------------------------------------------------  ---------  -----  --------  -----
silent+learned-spec vs random-actions                14.890     0.000  0.004     1.000
silent+learned-spec vs greedy+silent-centre          8.519      0.000  0.004     1.000
silent+learned-spec vs greedy+random-centre          11.454     0.000  0.004     1.000
silent+learned-spec vs no-comm                       7.117      0.000  0.004     1.000
silent+learned-spec vs oracle                        -27.886    0.000  0.004     0.000
hand-coded+learned-spec vs random-actions            34.564     0.000  0.004     1.000
hand-coded+learned-spec vs greedy+silent-centre      28.193     0.000  0.004     1.000
hand-coded+learned-spec vs greedy+random-centre      31.128     0.000  0.004     1.000
hand-coded+learned-spec vs no-comm                   26.791     0.000  0.004     1.000
hand-coded+learned-spec vs oracle                    -8.211     0.000  0.004     0.000
learned-centre+learned-spec vs random-actions        35.107     0.000  0.004     1.000
learned-centre+learned-spec vs greedy+silent-centre  28.736     0.000  0.004     1.000
learned-centre+learned-spec vs greedy+random-centre  31.672     0.000  0.004     1.000
learned-centre+learned-spec vs no-comm               27.335     0.000  0.004     1.000
learned-centre+learned-spec vs oracle                -7.668     0.000  0.004     0.000
greedy+hand-coded vs random-actions                  18.378     0.000  0.004     1.000
greedy+hand-coded vs greedy+silent-centre            12.007     0.000  0.004     1.000
greedy+hand-coded vs greedy+random-centre            14.942     0.000  0.004     1.000
greedy+hand-coded vs no-comm                         10.605     0.000  0.004     1.000
greedy+hand-coded vs oracle                          -24.398    0.000  0.004     0.000

system                       changed_random_matched  changed_random_matched_ci  changed_shuffle_batch  changed_shuffle_batch_ci  cic_bits  cic_bits_ci  distinct_codes  distinct_codes_ci  js_random_matched_bits  js_random_matched_bits_ci  js_shuffle_batch_bits  js_shuffle_batch_bits_ci  message_entropy_bits  message_entropy_bits_ci  mi_bits  mi_bits_ci  mi_direction_bits  mi_direction_bits_ci  mi_far_bits  mi_far_bits_ci  mi_null_q95_bits  mi_null_q95_bits_ci  mi_p_value  mi_p_value_ci  target_entropy_bits  target_entropy_bits_ci  topsim  topsim_ci  utilization  utilization_ci
---------------------------  ----------------------  -------------------------  ---------------------  ------------------------  --------  -----------  --------------  -----------------  ----------------------  -------------------------  ---------------------  ------------------------  --------------------  -----------------------  -------  ----------  -----------------  --------------------  -----------  --------------  ----------------  -------------------  ----------  -------------  -------------------  ----------------------  ------  ---------  -----------  --------------
silent+learned-spec          -                       -                          -                      -                         -         -            1.000           0.000              -                       -                          -                      -                         0.000                 0.000                    0.000    0.000       0.000              0.000                 0.000        0.000           0.000             0.000                1.000       0.000          2.441                0.058                   nan     nan        0.000        0.000         
hand-coded+learned-spec      0.242                   0.032                      0.242                  0.031                     0.489     0.048        8.000           0.000              0.195                   0.026                      0.193                  0.025                     2.654                 0.028                    2.654    0.028       1.731              0.018                 0.932        0.010           0.000             0.000                0.010       0.000          2.654                0.028                   0.740   0.007      0.885        0.009         
learned-centre+learned-spec  0.269                   0.014                      0.267                  0.016                     0.471     0.026        4.100           1.237              0.222                   0.011                      0.221                  0.012                     1.187                 0.180                    0.727    0.029       0.667              0.046                 0.021        0.016           0.000             0.000                0.010       0.000          2.596                0.036                   0.349   0.041      0.396        0.060         
greedy+hand-coded            0.411                   0.008                      0.413                  0.010                     0.854     0.006        8.000           0.000              0.400                   0.008                      0.401                  0.008                     2.631                 0.004                    2.631    0.004       1.907              0.003                 0.768        0.004           0.000             0.000                0.010       0.000          2.631                0.004                   0.746   0.004      0.877        0.001         
```
