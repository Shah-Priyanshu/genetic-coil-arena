# Squad: learned protocol vs hand-coded protocol at 3 bits (R6)

- Date: 2026-10-22
- Commit: 263d2f7 (code; this report is committed on top of it)
- Compute: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, 28 logical cores, 4 worker processes, numpy/numba only (no GPU). Training wall time 0.1 min (0.00 CPU-hours), evaluation 0.0 min.
- Training: 2 independent runs per condition and radius; 6 ES generations × population 8 × 2 episodes per member (≈0k episodes per run).
- Held-out evaluation: run s plays episodes 3,500,000,000 + s × 10,000 + i, i < 4. Training runs are the independent units for every CI and test (n = number of runs), so the CIs include training-seed variance, not only episode noise.
- Squad: 21×21 walls, 3 snakes, 2 food per colour, health 60, T = 300; channel = 1 symbol, V = 8 (3 bits) per receiver, sent every step (k = 1, the hand-coded centre's interval), read one step later.

Reproduce with `python scripts/learned_protocol.py --scale smoke --radii 3 --runs 2 --generations 6 --population 8 --episodes-per-member 2 --workers 4 --eval-episodes 4 --out docs/results/learned_protocol.md`.

## Design

- **Algorithm:** OpenAI-ES (`learning.EvolutionStrategy`: antithetic Gaussian perturbations σ = 0.05, centred-rank shaping, Adam lr = 0.03, weight decay 0.005). Centre and specialists are one parameter vector, so the joint system is optimised end-to-end from the team return through the discrete argmax channel, with no gradient through the message. All members of a generation play the same episodes and action noise. Fitness = team return (food + 0.01 per living snake per step). Same budget, sizes and hyperparameters for every condition; the three conditions of run s start from the same specialist weights.
- **Specialist (shared by the 3 snakes):** egocentric window planes (4·(2r+1)²), health, one-hot received code (8) → 16 tanh → 3 action logits, actions sampled from the softmax (3347 parameters at r = 3).
- **Learned centre (shared across receivers):** per receiver, its two own-colour food items as egocentric (ahead, right) / 21 + valid flag, sorted near to far, and its health (7 inputs) → 16 tanh → 8 logits; the code is the argmax (264 parameters). It gets food positions, not directions or distance classes: which partition of the plane to name is learned. Sorting by distance only removes a slot permutation.
- **Conditions:** (a) `silent+learned-spec`: the centre always sends code 0; (b) `hand-coded+learned-spec`: the fixed `HandCodedCentre` (`2·direction + far`); (c) `learned-centre+learned-spec`. Plus the scripted references `greedy+hand-coded` and the registered baselines (random actions, greedy + silent / random centre, no-comm greedy, full-view oracle).
- **Checkpoint:** the ES mean is scored every 3 generations on 32 fixed validation episodes (seeds 4,000,000,000 + run × 10,000 + i, disjoint from training and test); the best checkpoint is evaluated on the test seeds.
- **Metrics:** computed per training run and averaged (runs learn different codes, so messages are never pooled). T = the receiver's nearest own-colour food: direction relative to heading (4) × near/far (Manhattan > 5) + 'no food' (the hand-coded meaning). topsim = Spearman between Hamming distance of (direction, far) and of the code, on receivers with food. CIC from the specialists' exact softmax.

## Interpretation

- **r = 3:** team food (a) silent 0.25, (b) hand-coded 0.00, (c) learned 0.25; scripted greedy+hand-coded 15.12, no-comm 7.88, oracle 46.12. (c) − (a): +0.00 [-1.52, +1.52], not distinguishable, p(Holm) = 1.000. (c) − (b): +0.25 [-2.93, +3.43], not distinguishable, p(Holm) = 1.000. (b) − (a): -0.25 [-3.43, +2.93], not distinguishable, p(Holm) = 1.000. The learned centre uses 0.57 of 3 bits with MI(M;T) = 0.17 bits; CIC 0.001 bits (means over runs); interventions whose drop CI excludes 0: none.

<!-- discussion -->

## Team food per episode (held-out)

Mean ± 95% t-CI over training runs (each run's score is its mean over the test episodes; scripted systems and baselines use the same per-run episode sets).

| condition | role | r = 3 |
|---|---|---|
| silent+learned-spec | system | 0.25 ± 3.18 |
| hand-coded+learned-spec | system | 0.00 ± 0.00 |
| learned-centre+learned-spec | system | 0.25 ± 3.18 |
| greedy+hand-coded | system | 15.12 ± 52.41 |
| random-actions | baseline | 0.25 ± 0.00 |
| greedy+silent-centre | baseline | 9.25 ± 34.94 |
| greedy+random-centre | baseline | 4.12 ± 11.12 |
| no-comm | baseline | 7.88 ± 4.76 |
| oracle | baseline | 46.12 ± 42.88 |

## Headline comparisons

(c) vs (a): is learned communication useful at all; (c) vs (b): learned vs hand-coded protocol for learned listeners; (b) vs (a): do learned listeners use the hand-coded channel. Unpaired over training runs.

| r | comparison | mean diff [95% Welch CI] | p (Mann-Whitney, Holm) | A12 |
|---|---|---|---|---|
| 3 | learned-centre+learned-spec vs silent+learned-spec | 0.00 [-1.52, 1.52] | 1.000 | 0.50 |
| 3 | learned-centre+learned-spec vs hand-coded+learned-spec | 0.25 [-2.93, 3.43] | 1.000 | 0.75 |
| 3 | hand-coded+learned-spec vs silent+learned-spec | -0.25 [-3.43, 2.93] | 1.000 | 0.25 |
| 3 | learned-centre+learned-spec vs greedy+hand-coded | -14.88 [-66.49, 36.74] | 1.000 | 0.00 |
| 3 | hand-coded+learned-spec vs greedy+hand-coded | -15.12 [-67.54, 37.29] | 1.000 | 0.00 |

## Message ablations (learned systems)

Drop = intact − ablated team food, paired by training run; p = Wilcoxon signed-rank, Holm across the four interventions per system.

| r | system | intervention | intact | ablated | drop [95% CI] | p (Holm) | A12 | listening |
|---|---|---|---|---|---|---|---|---|
| 3 | silent+learned-spec | zero | 0.25 | 0.25 | 0.00 [0.00, 0.00] | 1.000 | 0.50 | no |
| 3 | silent+learned-spec | shuffle_batch | 0.25 | 0.25 | 0.00 [0.00, 0.00] | 1.000 | 0.50 | no |
| 3 | silent+learned-spec | swap_receivers | 0.25 | 0.25 | 0.00 [0.00, 0.00] | 1.000 | 0.50 | no |
| 3 | silent+learned-spec | random_matched | 0.25 | 0.25 | 0.00 [0.00, 0.00] | 1.000 | 0.50 | no |
| 3 | hand-coded+learned-spec | zero | 0.00 | 0.62 | -0.62 [-2.21, 0.96] | 1.000 | 0.00 | no |
| 3 | hand-coded+learned-spec | shuffle_batch | 0.00 | 0.12 | -0.12 [-1.71, 1.46] | 1.000 | 0.25 | no |
| 3 | hand-coded+learned-spec | swap_receivers | 0.00 | 0.00 | 0.00 [0.00, 0.00] | 1.000 | 0.50 | no |
| 3 | hand-coded+learned-spec | random_matched | 0.00 | 0.25 | -0.25 [-3.43, 2.93] | 1.000 | 0.25 | no |
| 3 | learned-centre+learned-spec | zero | 0.25 | 0.50 | -0.25 [-6.60, 6.10] | 1.000 | 0.25 | no |
| 3 | learned-centre+learned-spec | shuffle_batch | 0.25 | 0.00 | 0.25 [-2.93, 3.43] | 1.000 | 0.75 | no |
| 3 | learned-centre+learned-spec | swap_receivers | 0.25 | 0.12 | 0.12 [-1.46, 1.71] | 1.000 | 0.62 | no |
| 3 | learned-centre+learned-spec | random_matched | 0.25 | 0.00 | 0.25 [-2.93, 3.43] | 1.000 | 0.75 | no |

## What the centre says and how much it moves actions

Mean ± 95% CI over training runs; MI null q95 from 100 target shuffles (mean over runs).

| r | squad | H(M) bits | used / 3 bits | codes used | H(T) bits | MI(M;T) bits | MI null q95 | MI(M;direction) | MI(M;far) | topsim | CIC bits |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | silent+learned-spec | 0.000 ± 0.000 | 0% | 1.0 ± 0.0 | 2.693 ± 0.995 | 0.000 ± 0.000 | 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | n/a | n/a |
| 3 | hand-coded+learned-spec | 2.734 ± 0.574 | 91% | 8.0 ± 0.0 | 2.734 ± 0.574 | 2.777 ± 0.783 | 0.242 | 1.986 ± 0.061 | 0.785 ± 0.618 | 0.719 ± 0.236 | 0.005 ± 0.013 |
| 3 | learned-centre+learned-spec | 0.574 ± 1.098 | 19% | 3.0 ± 12.7 | 2.638 ± 1.285 | 0.166 ± 0.152 | 0.087 | 0.134 ± 0.003 | 0.040 ± 0.125 | -0.054 ± 0.489 | 0.001 ± 0.000 |
| 3 | greedy+hand-coded | 2.588 ± 0.897 | 86% | 7.5 ± 6.4 | 2.588 ± 0.897 | 2.596 ± 0.879 | 0.035 | 1.923 ± 0.251 | 0.722 ± 0.410 | 0.759 ± 0.216 | 0.875 ± 0.209 |

| r | squad | JS(shuffle_batch) bits | action changed (shuffle_batch) | JS(random_matched) bits | action changed (random_matched) |
|---|---|---|---|---|---|
| 3 | hand-coded+learned-spec | 0.001 ± 0.007 | 0.290 ± 0.696 | 0.002 ± 0.009 | 0.226 ± 0.631 |
| 3 | learned-centre+learned-spec | 0.002 ± 0.012 | 0.119 ± 1.513 | 0.002 ± 0.007 | 0.119 ± 0.303 |
| 3 | greedy+hand-coded | 0.353 ± 0.268 | 0.356 ± 0.292 | 0.349 ± 0.675 | 0.382 ± 0.848 |

### Learned centre, run by run

(a), (b), (c) = silent, hand-coded and learned centre with learned specialists trained from the same initial specialist weights.

| r | run | team food (c) | (b) | (a) | H(M) bits | codes used | MI(M;T) | MI(M;direction) | MI(M;far) | CIC bits | drop when zeroed | drop when shuffled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0 | 0.00 | 0.00 | 0.50 | 0.66 | 4 | 0.18 | 0.13 | 0.05 | 0.001 | -0.75 | 0.00 |
| 3 | 1 | 0.50 | 0.00 | 0.00 | 0.49 | 2 | 0.15 | 0.13 | 0.03 | 0.001 | 0.25 | 0.50 |

### Learned codebooks

r = 3, training run 0 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
| 4 | 7 | 14 | 8 | 24 | 9 | 25 | 6 | 19 | 0 |
| 6 | 0 | 0 | 0 | 1 | 0 | 8 | 0 | 1 | 0 |

r = 3, training run 1 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 1 | 0 | 11 | 0 | 2 | 0 | 0 | 0 |
| 7 | 12 | 16 | 2 | 15 | 5 | 39 | 3 | 26 | 0 |

## Learning curves

Full curves (every 3 generations, every run) in `learned_protocol_curves.csv`.

r = 3 (validation team food of the ES mean, mean over runs):

| condition | gen 0 | gen 3 | gen 6 | best ckpt gen (mean) |
|---|---|---|---|---|
| silent+learned-spec | 0.33 | 0.41 | 0.31 | 4 |
| hand-coded+learned-spec | 0.27 | 0.30 | 0.38 | 6 |
| learned-centre+learned-spec | 0.33 | 0.34 | 0.39 | 6 |

## Harness output

`evaluation.format_table` per radius.

### r = 3

```
Task: squad-learned-protocol-r3  (seeds: 2)
Baselines: random-actions, greedy+silent-centre, greedy+random-centre, no-comm, oracle

condition                    role      n  mean ± 95% CI    IQM [95% CI]           
---------------------------  --------  -  ---------------  -----------------------
silent+learned-spec          system    2  0.250 ± 3.177    0.250 [0.000, 0.500]   
hand-coded+learned-spec      system    2  0.000 ± 0.000    0.000 [0.000, 0.000]   
learned-centre+learned-spec  system    2  0.250 ± 3.177    0.250 [0.000, 0.500]   
greedy+hand-coded            system    2  15.125 ± 52.413  15.125 [11.000, 19.250]
random-actions               baseline  2  0.250 ± 0.000    0.250 [0.250, 0.250]   
greedy+silent-centre         baseline  2  9.250 ± 34.942   9.250 [6.500, 12.000]  
greedy+random-centre         baseline  2  4.125 ± 11.118   4.125 [3.250, 5.000]   
no-comm                      baseline  2  7.875 ± 4.765    7.875 [7.500, 8.250]   
oracle                       baseline  2  46.125 ± 42.883  46.125 [42.750, 49.500]

system                       intervention    ablated mean  drop [95% CI]           p (Holm)  A12    listening
---------------------------  --------------  ------------  ----------------------  --------  -----  ---------
silent+learned-spec          zero            0.250         0.000 [0.000, 0.000]    1.000     0.500  no       
silent+learned-spec          shuffle_batch   0.250         0.000 [0.000, 0.000]    1.000     0.500  no       
silent+learned-spec          swap_receivers  0.250         0.000 [0.000, 0.000]    1.000     0.500  no       
silent+learned-spec          random_matched  0.250         0.000 [0.000, 0.000]    1.000     0.500  no       
hand-coded+learned-spec      zero            0.625         -0.625 [-2.213, 0.963]  1.000     0.000  no       
hand-coded+learned-spec      shuffle_batch   0.125         -0.125 [-1.713, 1.463]  1.000     0.250  no       
hand-coded+learned-spec      swap_receivers  0.000         0.000 [0.000, 0.000]    1.000     0.500  no       
hand-coded+learned-spec      random_matched  0.250         -0.250 [-3.427, 2.927]  1.000     0.250  no       
learned-centre+learned-spec  zero            0.500         -0.250 [-6.603, 6.103]  1.000     0.250  no       
learned-centre+learned-spec  shuffle_batch   0.000         0.250 [-2.927, 3.427]   1.000     0.750  no       
learned-centre+learned-spec  swap_receivers  0.125         0.125 [-1.463, 1.713]   1.000     0.625  no       
learned-centre+learned-spec  random_matched  0.000         0.250 [-2.927, 3.427]   1.000     0.750  no       

comparison                                           mean diff  p      p (Holm)  A12  
---------------------------------------------------  ---------  -----  --------  -----
silent+learned-spec vs random-actions                0.000      1.000  1.000     0.500
silent+learned-spec vs greedy+silent-centre          -9.000     0.333  1.000     0.000
silent+learned-spec vs greedy+random-centre          -3.875     0.333  1.000     0.000
silent+learned-spec vs no-comm                       -7.625     0.333  1.000     0.000
silent+learned-spec vs oracle                        -45.875    0.333  1.000     0.000
hand-coded+learned-spec vs random-actions            -0.250     0.194  1.000     0.000
hand-coded+learned-spec vs greedy+silent-centre      -9.250     0.221  1.000     0.000
hand-coded+learned-spec vs greedy+random-centre      -4.125     0.221  1.000     0.000
hand-coded+learned-spec vs no-comm                   -7.875     0.221  1.000     0.000
hand-coded+learned-spec vs oracle                    -46.125    0.221  1.000     0.000
learned-centre+learned-spec vs random-actions        0.000      1.000  1.000     0.500
learned-centre+learned-spec vs greedy+silent-centre  -9.000     0.333  1.000     0.000
learned-centre+learned-spec vs greedy+random-centre  -3.875     0.333  1.000     0.000
learned-centre+learned-spec vs no-comm               -7.625     0.333  1.000     0.000
learned-centre+learned-spec vs oracle                -45.875    0.333  1.000     0.000
greedy+hand-coded vs random-actions                  14.875     0.221  1.000     1.000
greedy+hand-coded vs greedy+silent-centre            5.875      0.667  1.000     0.750
greedy+hand-coded vs greedy+random-centre            11.000     0.333  1.000     1.000
greedy+hand-coded vs no-comm                         7.250      0.333  1.000     1.000
greedy+hand-coded vs oracle                          -31.000    0.333  1.000     0.000

system                       changed_random_matched  changed_random_matched_ci  changed_shuffle_batch  changed_shuffle_batch_ci  cic_bits  cic_bits_ci  distinct_codes  distinct_codes_ci  js_random_matched_bits  js_random_matched_bits_ci  js_shuffle_batch_bits  js_shuffle_batch_bits_ci  message_entropy_bits  message_entropy_bits_ci  mi_bits  mi_bits_ci  mi_direction_bits  mi_direction_bits_ci  mi_far_bits  mi_far_bits_ci  mi_null_q95_bits  mi_null_q95_bits_ci  mi_p_value  mi_p_value_ci  target_entropy_bits  target_entropy_bits_ci  topsim  topsim_ci  utilization  utilization_ci
---------------------------  ----------------------  -------------------------  ---------------------  ------------------------  --------  -----------  --------------  -----------------  ----------------------  -------------------------  ---------------------  ------------------------  --------------------  -----------------------  -------  ----------  -----------------  --------------------  -----------  --------------  ----------------  -------------------  ----------  -------------  -------------------  ----------------------  ------  ---------  -----------  --------------
silent+learned-spec          -                       -                          -                      -                         -         -            1.000           0.000              -                       -                          -                      -                         0.000                 0.000                    0.000    0.000       0.000              0.000                 0.000        0.000           0.000             0.000                1.000       0.000          2.693                0.995                   nan     nan        0.000        0.000         
hand-coded+learned-spec      0.226                   0.631                      0.290                  0.696                     0.005     0.013        8.000           0.000              0.002                   0.009                      0.001                  0.007                     2.734                 0.574                    2.777    0.783       1.986              0.061                 0.785        0.618           0.242             1.267                0.010       0.000          2.734                0.574                   0.719   0.236      0.911        0.191         
learned-centre+learned-spec  0.119                   0.303                      0.119                  1.513                     0.001     0.000        3.000           12.706             0.002                   0.007                      0.002                  0.012                     0.574                 1.098                    0.166    0.152       0.134              0.003                 0.040        0.125           0.087             0.378                0.010       0.000          2.638                1.285                   -0.054  0.489      0.191        0.366         
greedy+hand-coded            0.382                   0.848                      0.356                  0.292                     0.875     0.209        7.500           6.353              0.349                   0.675                      0.353                  0.268                     2.588                 0.897                    2.596    0.879       1.923              0.251                 0.722        0.410           0.035             0.107                0.010       0.000          2.588                0.897                   0.759   0.216      0.863        0.299         
```
