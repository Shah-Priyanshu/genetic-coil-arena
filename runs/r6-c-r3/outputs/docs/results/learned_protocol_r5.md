# Squad: learned protocol vs hand-coded protocol at 3 bits (R6)

- Date: 2026-10-23
- Commit: 5ba43b8 (code; this report is committed on top of it)
- Compute: Intel(R) Xeon(R) Processor @ 2.80GHz, 4 logical cores, 4 worker processes, numpy/numba only (no GPU). Training wall time 25.2 min (1.58 CPU-hours), evaluation 0.4 min.
- Training: 4 independent runs per condition and radius; 500 ES generations × population 64 × 8 episodes per member (≈256k episodes per run).
- Held-out evaluation: run s plays episodes 3,500,000,000 + s × 10,000 + i, i < 100. Training runs are the independent units for every CI and test (n = number of runs), so the CIs include training-seed variance, not only episode noise.
- Squad: 21×21 walls, 3 snakes, 2 food per colour, health 60, T = 300; channel = 1 symbol, V = 8 (3 bits) per receiver, sent every step (k = 1, the hand-coded centre's interval), read one step later.

Reproduce with `python scripts/learned_protocol.py --scale cpu --radii 5 --runs 4 --generations 500 --population 64 --episodes-per-member 8 --workers 4 --eval-episodes 100 --out docs/results/learned_protocol_r5.md`.

## Design

- **Algorithm:** OpenAI-ES (`learning.EvolutionStrategy`: antithetic Gaussian perturbations σ = 0.05, centred-rank shaping, Adam lr = 0.03, weight decay 0.005). Centre and specialists are one parameter vector, so the joint system is optimised end-to-end from the team return through the discrete argmax channel, with no gradient through the message. All members of a generation play the same episodes and action noise. Fitness = team return (food + 0.01 per living snake per step). Same budget, sizes and hyperparameters for every condition; the three conditions of run s start from the same specialist weights.
- **Specialist (shared by the 3 snakes):** egocentric window planes (4·(2r+1)²), health, one-hot received code (8) → 16 tanh → 3 action logits, actions sampled from the softmax (7955 parameters at r = 5).
- **Learned centre (shared across receivers):** per receiver, its two own-colour food items as egocentric (ahead, right) / 21 + valid flag, sorted near to far, and its health (7 inputs) → 16 tanh → 8 logits; the code is the argmax (264 parameters). It gets food positions, not directions or distance classes: which partition of the plane to name is learned. Sorting by distance only removes a slot permutation.
- **Conditions:** (a) `silent+learned-spec`: the centre always sends code 0; (b) `hand-coded+learned-spec`: the fixed `HandCodedCentre` (`2·direction + far`); (c) `learned-centre+learned-spec`. Plus the scripted references `greedy+hand-coded` and the registered baselines (random actions, greedy + silent / random centre, no-comm greedy, full-view oracle).
- **Checkpoint:** the ES mean is scored every 10 generations on 32 fixed validation episodes (seeds 4,000,000,000 + run × 10,000 + i, disjoint from training and test); the best checkpoint is evaluated on the test seeds.
- **Metrics:** computed per training run and averaged (runs learn different codes, so messages are never pooled). T = the receiver's nearest own-colour food: direction relative to heading (4) × near/far (Manhattan > 5) + 'no food' (the hand-coded meaning). topsim = Spearman between Hamming distance of (direction, far) and of the code, on receivers with food. CIC from the specialists' exact softmax.

## Interpretation

- **r = 5:** team food (a) silent 2.92, (b) hand-coded 12.71, (c) learned 10.99; scripted greedy+hand-coded 26.64, no-comm 15.74, oracle 42.83. (c) − (a): +8.06 [-1.49, +17.62], not distinguishable, p(Holm) = 0.686. (c) − (b): -1.72 [-11.29, +7.86], not distinguishable, p(Holm) = 0.686. (b) − (a): +9.78 [+2.02, +17.55], higher, p(Holm) = 0.143. The learned centre uses 0.93 of 3 bits with MI(M;T) = 0.45 bits; CIC 0.242 bits (means over runs); interventions whose drop CI excludes 0: none.

### Discussion (hand-written)

This is a **reduced-budget** companion to `learned_protocol.md` (r = 3, 1000 generations). It has half the generations, and the network input is 2.4× larger (4·11² window pixels). Every learned condition is still climbing at generation 500, with best checkpoints at generations 420 to 500. The learned squads all sit below the scripted greedy listeners, so these numbers describe early learning, not final quality.

- The ordering matches r = 3. (b) hand-coded + learned listeners = 12.7, roughly equal to (c) learned centre = 11.0, and both are far above (a) silent = 2.9. (b) − (a) = +9.8 [+2.0, +17.6]; the (c) comparisons are inconclusive at n = 4.
- The silent learned squad has barely learned to forage from its window: 2.9 food, against 15.7 for no-comm greedy. A message is a shortcut that ES finds long before window-based foraging.
- In 3 of 4 runs a 1 to 1.4-bit direction code emerged (MI(M; direction) = 0.44 to 0.62 bits, MI(M; far) ≈ 0). Run 0 never formed a code (H(M) = 0.25 bits, MI = 0.04), the same failure mode as run 2 at r = 3.
- See `learned_protocol.md` for the full discussion and the statistics caveat. With 4 runs the smallest attainable Holm-corrected p-values are 0.143 and 0.5, so read the CIs.

## Team food per episode (held-out)

Mean ± 95% t-CI over training runs (each run's score is its mean over the test episodes; scripted systems and baselines use the same per-run episode sets).

| condition | role | r = 5 |
|---|---|---|
| silent+learned-spec | system | 2.92 ± 0.36 |
| hand-coded+learned-spec | system | 12.71 ± 7.77 |
| learned-centre+learned-spec | system | 10.99 ± 9.57 |
| greedy+hand-coded | system | 26.64 ± 1.47 |
| random-actions | baseline | 0.33 ± 0.13 |
| greedy+silent-centre | baseline | 14.91 ± 1.11 |
| greedy+random-centre | baseline | 7.43 ± 0.93 |
| no-comm | baseline | 15.74 ± 0.90 |
| oracle | baseline | 42.83 ± 0.46 |

## Headline comparisons

(c) vs (a): is learned communication useful at all; (c) vs (b): learned vs hand-coded protocol for learned listeners; (b) vs (a): do learned listeners use the hand-coded channel. Unpaired over training runs.

| r | comparison | mean diff [95% Welch CI] | p (Mann-Whitney, Holm) | A12 |
|---|---|---|---|---|
| 5 | learned-centre+learned-spec vs silent+learned-spec | 8.06 [-1.49, 17.62] | 0.686 | 0.75 |
| 5 | learned-centre+learned-spec vs hand-coded+learned-spec | -1.72 [-11.29, 7.86] | 0.686 | 0.38 |
| 5 | hand-coded+learned-spec vs silent+learned-spec | 9.78 [2.02, 17.55] | 0.143 | 1.00 |
| 5 | learned-centre+learned-spec vs greedy+hand-coded | -15.65 [-25.08, -6.21] | 0.143 | 0.00 |
| 5 | hand-coded+learned-spec vs greedy+hand-coded | -13.93 [-21.55, -6.31] | 0.143 | 0.00 |

## Message ablations (learned systems)

Drop = intact − ablated team food, paired by training run; p = Wilcoxon signed-rank, Holm across the four interventions per system.

| r | system | intervention | intact | ablated | drop [95% CI] | p (Holm) | A12 | listening |
|---|---|---|---|---|---|---|---|---|
| 5 | silent+learned-spec | zero | 2.92 | 2.92 | 0.00 [0.00, 0.00] | 1.000 | 0.50 | no |
| 5 | silent+learned-spec | shuffle_batch | 2.92 | 2.92 | 0.00 [0.00, 0.00] | 1.000 | 0.50 | no |
| 5 | silent+learned-spec | swap_receivers | 2.92 | 2.92 | 0.00 [0.00, 0.00] | 1.000 | 0.50 | no |
| 5 | silent+learned-spec | random_matched | 2.92 | 2.92 | 0.00 [0.00, 0.00] | 1.000 | 0.50 | no |
| 5 | hand-coded+learned-spec | zero | 12.71 | 1.87 | 10.84 [3.34, 18.34] | 0.500 | 1.00 | yes |
| 5 | hand-coded+learned-spec | shuffle_batch | 12.71 | 1.91 | 10.80 [2.64, 18.96] | 0.500 | 1.00 | yes |
| 5 | hand-coded+learned-spec | swap_receivers | 12.71 | 4.08 | 8.63 [1.77, 15.49] | 0.500 | 1.00 | yes |
| 5 | hand-coded+learned-spec | random_matched | 12.71 | 2.19 | 10.52 [2.12, 18.92] | 0.500 | 1.00 | yes |
| 5 | learned-centre+learned-spec | zero | 10.99 | 1.78 | 9.21 [-0.85, 19.26] | 0.500 | 1.00 | no |
| 5 | learned-centre+learned-spec | shuffle_batch | 10.99 | 1.81 | 9.18 [-1.17, 19.53] | 0.750 | 0.94 | no |
| 5 | learned-centre+learned-spec | swap_receivers | 10.99 | 3.94 | 7.05 [-0.99, 15.09] | 0.750 | 0.75 | no |
| 5 | learned-centre+learned-spec | random_matched | 10.99 | 2.01 | 8.98 [-1.16, 19.12] | 0.750 | 0.94 | no |

## What the centre says and how much it moves actions

Mean ± 95% CI over training runs; MI null q95 from 100 target shuffles (mean over runs).

| r | squad | H(M) bits | used / 3 bits | codes used | H(T) bits | MI(M;T) bits | MI null q95 | MI(M;direction) | MI(M;far) | topsim | CIC bits |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | silent+learned-spec | 0.000 ± 0.000 | 0% | 1.0 ± 0.0 | 2.628 ± 0.087 | 0.000 ± 0.000 | 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | n/a | n/a |
| 5 | hand-coded+learned-spec | 2.763 ± 0.119 | 92% | 8.0 ± 0.0 | 2.763 ± 0.119 | 2.764 ± 0.119 | 0.001 | 1.846 ± 0.075 | 0.928 ± 0.124 | 0.712 ± 0.040 | 0.346 ± 0.104 |
| 5 | learned-centre+learned-spec | 0.926 ± 0.766 | 31% | 3.0 ± 1.3 | 2.794 ± 0.200 | 0.452 ± 0.449 | 0.001 | 0.427 ± 0.436 | 0.011 ± 0.019 | 0.165 ± 0.177 | 0.242 ± 0.255 |
| 5 | greedy+hand-coded | 2.601 ± 0.015 | 87% | 8.0 ± 0.0 | 2.601 ± 0.015 | 2.601 ± 0.015 | 0.001 | 1.825 ± 0.011 | 0.827 ± 0.015 | 0.753 ± 0.011 | 0.652 ± 0.017 |

| r | squad | JS(shuffle_batch) bits | action changed (shuffle_batch) | JS(random_matched) bits | action changed (random_matched) |
|---|---|---|---|---|---|
| 5 | hand-coded+learned-spec | 0.165 ± 0.012 | 0.255 ± 0.038 | 0.166 ± 0.007 | 0.257 ± 0.045 |
| 5 | learned-centre+learned-spec | 0.110 ± 0.103 | 0.178 ± 0.145 | 0.113 ± 0.107 | 0.181 ± 0.145 |
| 5 | greedy+hand-coded | 0.299 ± 0.009 | 0.307 ± 0.012 | 0.306 ± 0.022 | 0.315 ± 0.026 |

### Learned centre, run by run

(a), (b), (c) = silent, hand-coded and learned centre with learned specialists trained from the same initial specialist weights.

| r | run | team food (c) | (b) | (a) | H(M) bits | codes used | MI(M;T) | MI(M;direction) | MI(M;far) | CIC bits | drop when zeroed | drop when shuffled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 0 | 2.45 | 5.74 | 2.71 | 0.25 | 2 | 0.04 | 0.04 | 0.00 | 0.002 | 0.32 | -0.05 |
| 5 | 1 | 14.55 | 12.93 | 3.19 | 1.39 | 4 | 0.64 | 0.62 | 0.01 | 0.323 | 12.63 | 12.96 |
| 5 | 2 | 15.76 | 15.72 | 2.76 | 1.00 | 3 | 0.51 | 0.44 | 0.03 | 0.337 | 14.60 | 14.41 |
| 5 | 3 | 11.20 | 16.44 | 3.04 | 1.05 | 3 | 0.62 | 0.61 | 0.00 | 0.304 | 9.28 | 9.39 |

### Learned codebooks

r = 5, training run 0 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 52 | 199 | 10 | 14 | 5 | 5 | 25 | 30 | 0 |
| 6 | 526 | 1503 | 426 | 2237 | 309 | 1652 | 297 | 791 | 0 |

r = 5, training run 1 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 2 | 157 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | 320 | 767 | 1 | 0 | 0 | 0 | 5 | 0 | 0 |
| 5 | 866 | 940 | 1206 | 1693 | 363 | 334 | 0 | 0 | 0 |
| 6 | 1026 | 924 | 8 | 0 | 576 | 746 | 1433 | 2168 | 0 |

r = 5, training run 2 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 387 | 0 | 220 | 246 | 1086 | 1009 | 2026 | 1572 | 0 |
| 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
| 6 | 2180 | 2689 | 82 | 270 | 0 | 0 | 298 | 1323 | 0 |

r = 5, training run 3 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 0 | 0 | 0 | 0 | 9 | 85 | 0 | 0 | 0 |
| 3 | 1550 | 2178 | 973 | 1199 | 229 | 322 | 58 | 9 | 0 |
| 7 | 362 | 378 | 4 | 0 | 668 | 962 | 1452 | 2721 | 0 |

## Learning curves

Full curves (every 10 generations, every run) in `learned_protocol_r5_curves.csv`.

r = 5 (validation team food of the ES mean, mean over runs):

| condition | gen 0 | gen 50 | gen 120 | gen 250 | gen 380 | gen 500 | best ckpt gen (mean) |
|---|---|---|---|---|---|---|---|
| silent+learned-spec | 0.32 | 0.70 | 1.22 | 2.20 | 2.80 | 3.00 | 422 |
| hand-coded+learned-spec | 0.31 | 0.93 | 2.00 | 6.04 | 10.28 | 13.25 | 488 |
| learned-centre+learned-spec | 0.30 | 0.94 | 1.46 | 3.98 | 8.58 | 11.34 | 490 |

## Harness output

`evaluation.format_table` per radius.

### r = 5

```
Task: squad-learned-protocol-r5  (seeds: 4)
Baselines: random-actions, greedy+silent-centre, greedy+random-centre, no-comm, oracle

condition                    role      n  mean ± 95% CI   IQM [95% CI]           
---------------------------  --------  -  --------------  -----------------------
silent+learned-spec          system    4  2.925 ± 0.364   2.900 [2.710, 3.190]   
hand-coded+learned-spec      system    4  12.707 ± 7.774  14.325 [5.740, 16.440] 
learned-centre+learned-spec  system    4  10.990 ± 9.565  12.875 [2.450, 15.760] 
greedy+hand-coded            system    4  26.638 ± 1.465  26.645 [25.570, 27.690]
random-actions               baseline  4  0.335 ± 0.133   0.295 [0.290, 0.460]   
greedy+silent-centre         baseline  4  14.905 ± 1.107  15.155 [13.890, 15.420]
greedy+random-centre         baseline  4  7.433 ± 0.930   7.620 [6.580, 7.910]   
no-comm                      baseline  4  15.738 ± 0.902  15.815 [15.000, 16.320]
oracle                       baseline  4  42.828 ± 0.464  42.905 [42.430, 43.070]

system                       intervention    ablated mean  drop [95% CI]           p (Holm)  A12    listening
---------------------------  --------------  ------------  ----------------------  --------  -----  ---------
silent+learned-spec          zero            2.925         0.000 [0.000, 0.000]    1.000     0.500  no       
silent+learned-spec          shuffle_batch   2.925         0.000 [0.000, 0.000]    1.000     0.500  no       
silent+learned-spec          swap_receivers  2.925         0.000 [0.000, 0.000]    1.000     0.500  no       
silent+learned-spec          random_matched  2.925         0.000 [0.000, 0.000]    1.000     0.500  no       
hand-coded+learned-spec      zero            1.868         10.840 [3.342, 18.338]  0.500     1.000  yes      
hand-coded+learned-spec      shuffle_batch   1.907         10.800 [2.641, 18.959]  0.500     1.000  yes      
hand-coded+learned-spec      swap_receivers  4.075         8.633 [1.772, 15.493]   0.500     1.000  yes      
hand-coded+learned-spec      random_matched  2.190         10.518 [2.119, 18.916]  0.500     1.000  yes      
learned-centre+learned-spec  zero            1.782         9.207 [-0.847, 19.262]  0.500     1.000  no       
learned-centre+learned-spec  shuffle_batch   1.812         9.178 [-1.171, 19.526]  0.750     0.938  no       
learned-centre+learned-spec  swap_receivers  3.940         7.050 [-0.991, 15.091]  0.750     0.750  no       
learned-centre+learned-spec  random_matched  2.010         8.980 [-1.161, 19.121]  0.750     0.938  no       

comparison                                           mean diff  p      p (Holm)  A12  
---------------------------------------------------  ---------  -----  --------  -----
silent+learned-spec vs random-actions                2.590      0.029  0.571     1.000
silent+learned-spec vs greedy+silent-centre          -11.980    0.029  0.571     0.000
silent+learned-spec vs greedy+random-centre          -4.508     0.029  0.571     0.000
silent+learned-spec vs no-comm                       -12.812    0.029  0.571     0.000
silent+learned-spec vs oracle                        -39.903    0.029  0.571     0.000
hand-coded+learned-spec vs random-actions            12.372     0.029  0.571     1.000
hand-coded+learned-spec vs greedy+silent-centre      -2.198     1.000  1.000     0.500
hand-coded+learned-spec vs greedy+random-centre      5.275      0.309  1.000     0.750
hand-coded+learned-spec vs no-comm                   -3.030     0.686  1.000     0.375
hand-coded+learned-spec vs oracle                    -30.120    0.029  0.571     0.000
learned-centre+learned-spec vs random-actions        10.655     0.029  0.571     1.000
learned-centre+learned-spec vs greedy+silent-centre  -3.915     0.486  1.000     0.312
learned-centre+learned-spec vs greedy+random-centre  3.557      0.309  1.000     0.750
learned-centre+learned-spec vs no-comm               -4.748     0.114  0.686     0.125
learned-centre+learned-spec vs oracle                -31.838    0.029  0.571     0.000
greedy+hand-coded vs random-actions                  26.303     0.029  0.571     1.000
greedy+hand-coded vs greedy+silent-centre            11.733     0.029  0.571     1.000
greedy+hand-coded vs greedy+random-centre            19.205     0.029  0.571     1.000
greedy+hand-coded vs no-comm                         10.900     0.029  0.571     1.000
greedy+hand-coded vs oracle                          -16.190    0.029  0.571     0.000

system                       changed_random_matched  changed_random_matched_ci  changed_shuffle_batch  changed_shuffle_batch_ci  cic_bits  cic_bits_ci  distinct_codes  distinct_codes_ci  js_random_matched_bits  js_random_matched_bits_ci  js_shuffle_batch_bits  js_shuffle_batch_bits_ci  message_entropy_bits  message_entropy_bits_ci  mi_bits  mi_bits_ci  mi_direction_bits  mi_direction_bits_ci  mi_far_bits  mi_far_bits_ci  mi_null_q95_bits  mi_null_q95_bits_ci  mi_p_value  mi_p_value_ci  target_entropy_bits  target_entropy_bits_ci  topsim  topsim_ci  utilization  utilization_ci
---------------------------  ----------------------  -------------------------  ---------------------  ------------------------  --------  -----------  --------------  -----------------  ----------------------  -------------------------  ---------------------  ------------------------  --------------------  -----------------------  -------  ----------  -----------------  --------------------  -----------  --------------  ----------------  -------------------  ----------  -------------  -------------------  ----------------------  ------  ---------  -----------  --------------
silent+learned-spec          -                       -                          -                      -                         -         -            1.000           0.000              -                       -                          -                      -                         0.000                 0.000                    0.000    0.000       0.000              0.000                 0.000        0.000           0.000             0.000                1.000       0.000          2.628                0.087                   nan     nan        0.000        0.000         
hand-coded+learned-spec      0.257                   0.045                      0.255                  0.038                     0.346     0.104        8.000           0.000              0.166                   0.007                      0.165                  0.012                     2.763                 0.119                    2.764    0.119       1.846              0.075                 0.928        0.124           0.001             0.001                0.010       0.000          2.763                0.119                   0.712   0.040      0.921        0.040         
learned-centre+learned-spec  0.181                   0.145                      0.178                  0.145                     0.242     0.255        3.000           1.299              0.113                   0.107                      0.110                  0.103                     0.926                 0.766                    0.452    0.449       0.427              0.436                 0.011        0.019           0.001             0.000                0.010       0.000          2.794                0.200                   0.165   0.177      0.309        0.255         
greedy+hand-coded            0.315                   0.026                      0.307                  0.012                     0.652     0.017        8.000           0.000              0.306                   0.022                      0.299                  0.009                     2.601                 0.015                    2.601    0.015       1.825              0.011                 0.827        0.015           0.001             0.000                0.010       0.000          2.601                0.015                   0.753   0.011      0.867        0.005         
```
