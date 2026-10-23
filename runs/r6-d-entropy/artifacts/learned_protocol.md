# Squad: learned protocol vs hand-coded protocol at 3 bits (R6)

- Date: 2026-10-23
- Commit: 3f3ab58 (code; this report is committed on top of it)
- Compute: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, 28 logical cores, 20 worker processes, numpy/numba only (no GPU). Training wall time 74.2 min (12.23 CPU-hours), evaluation 0.3 min.
- Training: 10 independent runs per condition and radius; 2000 ES generations × population 128 × 8 episodes per member (≈2048k episodes per run).
- Held-out evaluation: run s plays episodes 3,500,000,000 + s × 10,000 + i, i < 200. Training runs are the independent units for every CI and test (n = number of runs), so the CIs include training-seed variance, not only episode noise.
- Squad: 21×21 walls, 3 snakes, 2 food per colour, health 60, T = 300; channel = 1 symbol, V = 8 (3 bits) per receiver, sent every step (k = 1, the hand-coded centre's interval), read one step later.

Reproduce with `python scripts/learned_protocol.py --scale gpu --radii 3 --runs 10 --generations 2000 --population 128 --episodes-per-member 8 --workers 20 --entropy-bonus 1 --entropy-anneal 0.15 --eval-episodes 200 --out docs/results/learned_protocol.md`.

## Design

- **Algorithm:** OpenAI-ES (`learning.EvolutionStrategy`: antithetic Gaussian perturbations σ = 0.05, centred-rank shaping, Adam lr = 0.03, weight decay 0.005). Centre and specialists are one parameter vector, so the joint system is optimised end-to-end from the team return through the discrete argmax channel, with no gradient through the message. All members of a generation play the same episodes and action noise. Fitness = team return (food + 0.01 per living snake per step). Same budget, sizes and hyperparameters for every condition; the three conditions of run s start from the same specialist weights.
- **Specialist (shared by the 3 snakes):** egocentric window planes (4·(2r+1)²), health, one-hot received code (8) → 16 tanh → 3 action logits, actions sampled from the softmax (3611 parameters at r = 3).
- **Learned centre (shared across receivers):** per receiver, its two own-colour food items as egocentric (ahead, right) / 21 + valid flag, sorted near to far, and its health (7 inputs) → 16 tanh → 8 logits; the code is the argmax (3611 parameters). It gets food positions, not directions or distance classes: which partition of the plane to name is learned. Sorting by distance only removes a slot permutation.
- **Conditions:** (a) `silent+learned-spec`: the centre always sends code 0; (b) `hand-coded+learned-spec`: the fixed `HandCodedCentre` (`2·direction + far`); (c) `learned-centre+learned-spec`. Plus the scripted references `greedy+hand-coded` and the registered baselines (random actions, greedy + silent / random centre, no-comm greedy, full-view oracle).
- **Checkpoint:** the ES mean is scored every 10 generations on 32 fixed validation episodes (seeds 4,000,000,000 + run × 10,000 + i, disjoint from training and test); the best checkpoint is evaluated on the test seeds.
- **Metrics:** computed per training run and averaged (runs learn different codes, so messages are never pooled). T = the receiver's nearest own-colour food: direction relative to heading (4) × near/far (Manhattan > 5) + 'no food' (the hand-coded meaning). topsim = Spearman between Hamming distance of (direction, far) and of the code, on receivers with food. CIC from the specialists' exact softmax.

## Interpretation

- **r = 3:** team food (a) silent nan, (b) hand-coded nan, (c) learned 36.05; scripted greedy+hand-coded 18.70, no-comm 8.10, oracle 43.10. (c) − (a): n/a. (c) − (b): n/a. (b) − (a): n/a. The learned centre uses 1.31 of 3 bits with MI(M;T) = 0.69 bits; CIC 0.497 bits (means over runs); interventions whose drop CI excludes 0: `zero`, `shuffle_batch`, `swap_receivers`, `random_matched`.

<!-- discussion -->

## Team food per episode (held-out)

Mean ± 95% t-CI over training runs (each run's score is its mean over the test episodes; scripted systems and baselines use the same per-run episode sets).

| condition | role | r = 3 |
|---|---|---|
| learned-centre+learned-spec | system | 36.05 ± 2.00 |
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
| 3 | learned-centre+learned-spec vs greedy+hand-coded | 17.34 [15.33, 19.36] | 0.00018 | 1.00 |

## Message ablations (learned systems)

Drop = intact − ablated team food, paired by training run; p = Wilcoxon signed-rank, Holm across the four interventions per system.

| r | system | intervention | intact | ablated | drop [95% CI] | p (Holm) | A12 | listening |
|---|---|---|---|---|---|---|---|---|
| 3 | learned-centre+learned-spec | zero | 36.05 | 2.82 | 33.23 [30.80, 35.65] | 0.0078 | 1.00 | yes |
| 3 | learned-centre+learned-spec | shuffle_batch | 36.05 | 2.44 | 33.60 [31.56, 35.64] | 0.0078 | 1.00 | yes |
| 3 | learned-centre+learned-spec | swap_receivers | 36.05 | 9.76 | 26.28 [24.15, 28.41] | 0.0078 | 1.00 | yes |
| 3 | learned-centre+learned-spec | random_matched | 36.05 | 2.67 | 33.38 [31.25, 35.50] | 0.0078 | 1.00 | yes |

## What the centre says and how much it moves actions

Mean ± 95% CI over training runs; MI null q95 from 100 target shuffles (mean over runs).

| r | squad | H(M) bits | used / 3 bits | codes used | H(T) bits | MI(M;T) bits | MI null q95 | MI(M;direction) | MI(M;far) | topsim | CIC bits |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | learned-centre+learned-spec | 1.312 ± 0.110 | 44% | 4.5 ± 0.8 | 2.616 ± 0.056 | 0.690 ± 0.061 | 0.000 | 0.623 ± 0.060 | 0.020 ± 0.011 | 0.276 ± 0.019 | 0.497 ± 0.034 |
| 3 | greedy+hand-coded | 2.631 ± 0.004 | 88% | 8.0 ± 0.0 | 2.631 ± 0.004 | 2.631 ± 0.004 | 0.000 | 1.907 ± 0.003 | 0.768 ± 0.004 | 0.746 ± 0.004 | 0.854 ± 0.006 |

| r | squad | JS(shuffle_batch) bits | action changed (shuffle_batch) | JS(random_matched) bits | action changed (random_matched) |
|---|---|---|---|---|---|
| 3 | learned-centre+learned-spec | 0.225 ± 0.011 | 0.269 ± 0.016 | 0.224 ± 0.011 | 0.268 ± 0.016 |
| 3 | greedy+hand-coded | 0.401 ± 0.008 | 0.413 ± 0.010 | 0.400 ± 0.008 | 0.411 ± 0.008 |

### Learned centre, run by run

(a), (b), (c) = silent, hand-coded and learned centre with learned specialists trained from the same initial specialist weights.

| r | run | team food (c) | (b) | (a) | H(M) bits | codes used | MI(M;T) | MI(M;direction) | MI(M;far) | CIC bits | drop when zeroed | drop when shuffled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0 | 32.23 | nan | nan | 1.31 | 3 | 0.83 | 0.79 | 0.02 | 0.393 | 27.53 | 28.62 |
| 3 | 1 | 37.08 | nan | nan | 1.35 | 6 | 0.67 | 0.54 | 0.03 | 0.496 | 35.88 | 34.85 |
| 3 | 2 | 38.87 | nan | nan | 1.43 | 4 | 0.73 | 0.59 | 0.05 | 0.518 | 35.70 | 35.66 |
| 3 | 3 | 36.52 | nan | nan | 1.13 | 4 | 0.58 | 0.56 | 0.02 | 0.484 | 30.99 | 35.19 |
| 3 | 4 | 36.30 | nan | nan | 1.35 | 5 | 0.64 | 0.61 | 0.01 | 0.484 | 35.15 | 33.65 |
| 3 | 5 | 38.95 | nan | nan | 1.63 | 5 | 0.74 | 0.65 | 0.04 | 0.496 | 36.26 | 36.27 |
| 3 | 6 | 36.09 | nan | nan | 1.34 | 4 | 0.72 | 0.66 | 0.02 | 0.506 | 34.91 | 33.30 |
| 3 | 7 | 38.46 | nan | nan | 1.29 | 5 | 0.73 | 0.64 | 0.01 | 0.581 | 36.56 | 36.41 |
| 3 | 8 | 30.38 | nan | nan | 1.21 | 3 | 0.72 | 0.70 | 0.00 | 0.478 | 29.25 | 28.59 |
| 3 | 9 | 35.59 | nan | nan | 1.08 | 6 | 0.55 | 0.50 | 0.01 | 0.535 | 30.03 | 33.51 |

### Learned codebooks

r = 3, training run 0 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 7615 | 11625 | 1813 | 474 | 29 | 0 | 1856 | 3956 | 0 |
| 6 | 20 | 68 | 909 | 4004 | 41 | 336 | 0 | 0 | 0 |
| 7 | 0 | 0 | 137 | 10 | 1131 | 3561 | 1961 | 4848 | 0 |

r = 3, training run 1 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1603 | 0 | 361 | 149 | 1147 | 3553 | 3934 | 5124 | 0 |
| 2 | 6521 | 12303 | 32 | 513 | 0 | 0 | 533 | 4949 | 0 |
| 3 | 88 | 0 | 7 | 39 | 0 | 0 | 5 | 0 | 0 |
| 4 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 4 | 0 |
| 6 | 7 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| 7 | 1277 | 12 | 747 | 1871 | 59 | 111 | 74 | 0 | 0 |

r = 3, training run 2 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 2880 | 762 | 994 | 2272 | 7 | 20 | 0 | 0 | 0 |
| 1 | 5270 | 11798 | 2 | 17 | 0 | 0 | 291 | 7013 | 0 |
| 2 | 1293 | 0 | 132 | 94 | 876 | 3882 | 3799 | 5010 | 0 |
| 3 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

r = 3, training run 3 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 2 | 0 | 56 | 1650 | 14 | 14 | 13 | 26 | 0 |
| 3 | 0 | 0 | 1 | 4 | 0 | 2 | 0 | 0 | 0 |
| 4 | 9 | 0 | 3128 | 6102 | 1190 | 4512 | 735 | 1180 | 0 |
| 6 | 9777 | 12080 | 1080 | 5543 | 1 | 0 | 519 | 1404 | 0 |

r = 3, training run 4 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 379 | 0 | 112 | 21 | 947 | 3196 | 2615 | 5199 | 0 |
| 1 | 7769 | 10778 | 17 | 0 | 13 | 0 | 2382 | 6499 | 0 |
| 3 | 759 | 1015 | 1013 | 1977 | 136 | 268 | 27 | 0 | 0 |
| 4 | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 | 163 | 0 | 27 | 0 | 6 | 2 | 0 | 0 | 0 |

r = 3, training run 5 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 2570 | 984 | 880 | 2158 | 22 | 22 | 41 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 | 0 | 46 | 5 | 1472 | 0 |
| 3 | 973 | 0 | 124 | 183 | 964 | 3578 | 3715 | 4621 | 0 |
| 5 | 6045 | 11618 | 0 | 0 | 0 | 0 | 998 | 4897 | 0 |
| 7 | 1 | 4 | 0 | 0 | 0 | 0 | 0 | 289 | 0 |

r = 3, training run 6 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 761 | 0 | 297 | 400 | 1118 | 3714 | 3902 | 7059 | 0 |
| 1 | 8333 | 10322 | 148 | 0 | 3 | 0 | 696 | 3348 | 0 |
| 3 | 69 | 1483 | 363 | 1591 | 1 | 4 | 0 | 0 | 0 |
| 6 | 56 | 0 | 20 | 11 | 8 | 0 | 7 | 56 | 0 |

r = 3, training run 7 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 2 | 0 | 6 | 8 | 2 | 0 | 0 | 0 | 0 |
| 3 | 786 | 40 | 0 | 0 | 0 | 0 | 906 | 1924 | 0 |
| 4 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| 5 | 7913 | 12224 | 457 | 4838 | 0 | 0 | 1 | 227 | 0 |
| 7 | 1132 | 0 | 3490 | 4930 | 1119 | 3740 | 366 | 370 | 0 |

r = 3, training run 8 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 10 | 0 | 441 | 3 | 1491 | 3554 | 1243 | 3188 | 0 |
| 3 | 7242 | 11471 | 860 | 1106 | 11 | 0 | 3427 | 6041 | 0 |
| 4 | 1 | 0 | 1499 | 2908 | 117 | 249 | 0 | 0 | 0 |

r = 3, training run 9 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 156 | 0 | 0 | 0 | 0 | 0 |
| 1 | 804 | 0 | 3832 | 6575 | 1025 | 4643 | 571 | 1132 | 0 |
| 2 | 171 | 0 | 68 | 69 | 1 | 0 | 23 | 38 | 0 |
| 3 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| 6 | 0 | 0 | 0 | 100 | 0 | 0 | 0 | 0 | 0 |
| 7 | 8279 | 12458 | 509 | 4341 | 0 | 0 | 315 | 1406 | 0 |

## Learning curves

Full curves (every 10 generations, every run) in `learned_protocol_curves.csv`.

r = 3 (validation team food of the ES mean, mean over runs):

| condition | gen 0 | gen 200 | gen 500 | gen 1000 | gen 1500 | gen 2000 | best ckpt gen (mean) |
|---|---|---|---|---|---|---|---|
| learned-centre+learned-spec | 0.30 | 15.81 | 30.62 | 34.40 | 35.74 | 36.13 | 1759 |

## Harness output

`evaluation.format_table` per radius.

### r = 3

```
Task: squad-learned-protocol-r3  (seeds: 10)
Baselines: random-actions, greedy+silent-centre, greedy+random-centre, no-comm, oracle

condition                    role      n   mean ± 95% CI   IQM [95% CI]           
---------------------------  --------  --  --------------  -----------------------
learned-centre+learned-spec  system    10  36.046 ± 2.002  36.672 [34.182, 37.988]
greedy+hand-coded            system    10  18.703 ± 0.325  18.724 [18.336, 19.061]
random-actions               baseline  10  0.325 ± 0.027   0.321 [0.302, 0.347]   
greedy+silent-centre         baseline  10  6.696 ± 0.217   6.701 [6.495, 6.903]   
greedy+random-centre         baseline  10  3.761 ± 0.162   3.732 [3.593, 3.926]   
no-comm                      baseline  10  8.098 ± 0.272   8.102 [7.856, 8.354]   
oracle                       baseline  10  43.100 ± 0.315  43.145 [42.829, 43.393]

system                       intervention    ablated mean  drop [95% CI]            p (Holm)  A12    listening
---------------------------  --------------  ------------  -----------------------  --------  -----  ---------
learned-centre+learned-spec  zero            2.818         33.228 [30.802, 35.653]  0.008     1.000  yes      
learned-centre+learned-spec  shuffle_batch   2.443         33.603 [31.562, 35.644]  0.008     1.000  yes      
learned-centre+learned-spec  swap_receivers  9.765         26.282 [24.149, 28.414]  0.008     1.000  yes      
learned-centre+learned-spec  random_matched  2.671         33.375 [31.253, 35.497]  0.008     1.000  yes      

comparison                                           mean diff  p      p (Holm)  A12  
---------------------------------------------------  ---------  -----  --------  -----
learned-centre+learned-spec vs random-actions        35.721     0.000  0.002     1.000
learned-centre+learned-spec vs greedy+silent-centre  29.351     0.000  0.002     1.000
learned-centre+learned-spec vs greedy+random-centre  32.286     0.000  0.002     1.000
learned-centre+learned-spec vs no-comm               27.949     0.000  0.002     1.000
learned-centre+learned-spec vs oracle                -7.054     0.000  0.002     0.000
greedy+hand-coded vs random-actions                  18.378     0.000  0.002     1.000
greedy+hand-coded vs greedy+silent-centre            12.007     0.000  0.002     1.000
greedy+hand-coded vs greedy+random-centre            14.942     0.000  0.002     1.000
greedy+hand-coded vs no-comm                         10.605     0.000  0.002     1.000
greedy+hand-coded vs oracle                          -24.398    0.000  0.002     0.000

system                       changed_random_matched  changed_random_matched_ci  changed_shuffle_batch  changed_shuffle_batch_ci  cic_bits  cic_bits_ci  distinct_codes  distinct_codes_ci  js_random_matched_bits  js_random_matched_bits_ci  js_shuffle_batch_bits  js_shuffle_batch_bits_ci  message_entropy_bits  message_entropy_bits_ci  mi_bits  mi_bits_ci  mi_direction_bits  mi_direction_bits_ci  mi_far_bits  mi_far_bits_ci  mi_null_q95_bits  mi_null_q95_bits_ci  mi_p_value  mi_p_value_ci  target_entropy_bits  target_entropy_bits_ci  topsim  topsim_ci  utilization  utilization_ci
---------------------------  ----------------------  -------------------------  ---------------------  ------------------------  --------  -----------  --------------  -----------------  ----------------------  -------------------------  ---------------------  ------------------------  --------------------  -----------------------  -------  ----------  -----------------  --------------------  -----------  --------------  ----------------  -------------------  ----------  -------------  -------------------  ----------------------  ------  ---------  -----------  --------------
learned-centre+learned-spec  0.268                   0.016                      0.269                  0.016                     0.497     0.034        4.500           0.773              0.224                   0.011                      0.225                  0.011                     1.312                 0.110                    0.690    0.061       0.623              0.060                 0.020        0.011           0.000             0.000                0.010       0.000          2.616                0.056                   0.276   0.019      0.437        0.037         
greedy+hand-coded            0.411                   0.008                      0.413                  0.010                     0.854     0.006        8.000           0.000              0.400                   0.008                      0.401                  0.008                     2.631                 0.004                    2.631    0.004       1.907              0.003                 0.768        0.004           0.000             0.000                0.010       0.000          2.631                0.004                   0.746   0.004      0.877        0.001         
```
