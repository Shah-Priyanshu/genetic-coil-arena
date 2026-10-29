# Squad: learned protocol vs hand-coded protocol at 3 bits (R6)

- Date: 2026-10-29
- Commit: 0606bbaf (code; this report is committed on top of it)
- Compute: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel, 28 logical cores, 1 worker processes, numpy/numba only (no GPU). Training wall time 0.0 min (0.00 CPU-hours), evaluation 0.0 min.
- Training: 1 independent runs per condition and radius; 1 ES generations × population 2 × 1 episodes per member (≈0k episodes per run).
- Held-out evaluation: run s plays episodes 3,500,000,000 + s × 10,000 + i, i < 2. Training runs are the independent units for every CI and test (n = number of runs), so the CIs include training-seed variance, not only episode noise.
- Squad: 21×21 walls, 3 snakes, 2 food per colour, health 60, T = 300; channel = 1 symbol, V = 8 (3 bits) per receiver, sent every step (k = 1, the hand-coded centre's interval), read one step later.

Reproduce with `python scripts/learned_protocol.py --scale smoke --radii 3 --runs 1 --generations 1 --population 2 --episodes-per-member 1 --workers 1 --eval-episodes 2 --out docs/results/lp.md`.

## Design

- **Algorithm:** OpenAI-ES (`learning.EvolutionStrategy`: antithetic Gaussian perturbations σ = 0.05, centred-rank shaping, Adam lr = 0.03, weight decay 0.005). Centre and specialists are one parameter vector, so the joint system is optimised end-to-end from the team return through the discrete argmax channel, with no gradient through the message. All members of a generation play the same episodes and action noise. Fitness = team return (food + 0.01 per living snake per step). Same budget, sizes and hyperparameters for every condition; the three conditions of run s start from the same specialist weights.
- **Specialist (shared by the 3 snakes):** egocentric window planes (4·(2r+1)²), health, one-hot received code (8) → 16 tanh → 3 action logits, actions sampled from the softmax (3347 parameters at r = 3).
- **Learned centre (shared across receivers):** per receiver, its two own-colour food items as egocentric (ahead, right) / 21 + valid flag, sorted near to far, and its health (7 inputs) → 16 tanh → 8 logits; the code is the argmax (-3347 parameters). It gets food positions, not directions or distance classes: which partition of the plane to name is learned. Sorting by distance only removes a slot permutation.
- **Conditions:** (a) `silent+learned-spec`: the centre always sends code 0; (b) `hand-coded+learned-spec`: the fixed `HandCodedCentre` (`2·direction + far`); (c) `learned-centre+learned-spec`. Plus the scripted references `greedy+hand-coded` and the registered baselines (random actions, greedy + silent / random centre, no-comm greedy, full-view oracle).
- **Checkpoint:** the ES mean is scored every 1 generations on 32 fixed validation episodes (seeds 4,000,000,000 + run × 10,000 + i, disjoint from training and test); the best checkpoint is evaluated on the test seeds.
- **Metrics:** computed per training run and averaged (runs learn different codes, so messages are never pooled). T = the receiver's nearest own-colour food: direction relative to heading (4) × near/far (Manhattan > 5) + 'no food' (the hand-coded meaning). topsim = Spearman between Hamming distance of (direction, far) and of the code, on receivers with food. CIC from the specialists' exact softmax.

## Interpretation

- **r = 3:** team food (a) silent 0.00, (b) hand-coded nan, (c) learned nan; scripted greedy+hand-coded 19.00, no-comm 12.50, oracle 47.50. (c) − (a): n/a. (c) − (b): n/a. (b) − (a): n/a. The learned centre uses nan of 3 bits with MI(M;T) = nan bits; CIC nan bits (means over runs); interventions whose drop CI excludes 0: none.

<!-- discussion -->

## Team food per episode (held-out)

Mean ± 95% t-CI over training runs (each run's score is its mean over the test episodes; scripted systems and baselines use the same per-run episode sets).

| condition | role | r = 3 |
|---|---|---|
| silent+learned-spec | system | 0.00 ± nan |
| greedy+hand-coded | system | 19.00 ± nan |
| random-actions | baseline | 0.00 ± nan |
| greedy+silent-centre | baseline | 12.00 ± nan |
| greedy+random-centre | baseline | 4.00 ± nan |
| no-comm | baseline | 12.50 ± nan |
| oracle | baseline | 47.50 ± nan |

## Headline comparisons

(c) vs (a): is learned communication useful at all; (c) vs (b): learned vs hand-coded protocol for learned listeners; (b) vs (a): do learned listeners use the hand-coded channel. Unpaired over training runs.

| r | comparison | mean diff [95% Welch CI] | p (Mann-Whitney, Holm) | A12 |
|---|---|---|---|---|

## Message ablations (learned systems)

Drop = intact − ablated team food, paired by training run; p = Wilcoxon signed-rank, Holm across the four interventions per system.

| r | system | intervention | intact | ablated | drop [95% CI] | p (Holm) | A12 | listening |
|---|---|---|---|---|---|---|---|---|
| 3 | silent+learned-spec | zero | 0.00 | 0.00 | 0.00 [nan, nan] | 1.000 | 0.50 | no |
| 3 | silent+learned-spec | shuffle_batch | 0.00 | 0.00 | 0.00 [nan, nan] | 1.000 | 0.50 | no |
| 3 | silent+learned-spec | swap_receivers | 0.00 | 0.00 | 0.00 [nan, nan] | 1.000 | 0.50 | no |
| 3 | silent+learned-spec | random_matched | 0.00 | 0.00 | 0.00 [nan, nan] | 1.000 | 0.50 | no |

## What the centre says and how much it moves actions

Mean ± 95% CI over training runs; MI null q95 from 100 target shuffles (mean over runs).

| r | squad | H(M) bits | used / 3 bits | codes used | H(T) bits | MI(M;T) bits | MI null q95 | MI(M;direction) | MI(M;far) | topsim | CIC bits |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | silent+learned-spec | 0.000 | 0% | 1.0 | 2.642 | 0.000 | 0.000 | 0.000 | 0.000 | n/a | n/a |
| 3 | greedy+hand-coded | 2.491 | 83% | 7.0 | 2.491 | 2.504 | 0.052 | 1.843 | 0.725 | 0.777 | 0.755 |

| r | squad | JS(shuffle_batch) bits | action changed (shuffle_batch) | JS(random_matched) bits | action changed (random_matched) |
|---|---|---|---|---|---|
| 3 | greedy+hand-coded | 0.289 | 0.300 | 0.333 | 0.325 |

### Learned centre, run by run

(a), (b), (c) = silent, hand-coded and learned centre with learned specialists trained from the same initial specialist weights.

| r | run | team food (c) | (b) | (a) | H(M) bits | codes used | MI(M;T) | MI(M;direction) | MI(M;far) | CIC bits | drop when zeroed | drop when shuffled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

### Learned codebooks

## Learning curves

Full curves (every 1 generations, every run) in `lp_curves.csv`.

r = 3 (validation team food of the ES mean, mean over runs):

| condition | gen 0 | gen 1 | best ckpt gen (mean) |
|---|---|---|---|
| silent+learned-spec | 0.38 | 0.44 | 1 |

## Harness output

`evaluation.format_table` per radius.

### r = 3

```
Task: squad-learned-protocol-r3  (seeds: 1)
Baselines: random-actions, greedy+silent-centre, greedy+random-centre, no-comm, oracle

condition             role      n  mean ± 95% CI  IQM [95% CI]     
--------------------  --------  -  -------------  -----------------
silent+learned-spec   system    1  0.000 ± nan    0.000 [nan, nan] 
greedy+hand-coded     system    1  19.000 ± nan   19.000 [nan, nan]
random-actions        baseline  1  0.000 ± nan    0.000 [nan, nan] 
greedy+silent-centre  baseline  1  12.000 ± nan   12.000 [nan, nan]
greedy+random-centre  baseline  1  4.000 ± nan    4.000 [nan, nan] 
no-comm               baseline  1  12.500 ± nan   12.500 [nan, nan]
oracle                baseline  1  47.500 ± nan   47.500 [nan, nan]

system               intervention    ablated mean  drop [95% CI]     p (Holm)  A12    listening
-------------------  --------------  ------------  ----------------  --------  -----  ---------
silent+learned-spec  zero            0.000         0.000 [nan, nan]  1.000     0.500  no       
silent+learned-spec  shuffle_batch   0.000         0.000 [nan, nan]  1.000     0.500  no       
silent+learned-spec  swap_receivers  0.000         0.000 [nan, nan]  1.000     0.500  no       
silent+learned-spec  random_matched  0.000         0.000 [nan, nan]  1.000     0.500  no       

comparison                                   mean diff  p      p (Holm)  A12  
-------------------------------------------  ---------  -----  --------  -----
silent+learned-spec vs random-actions        0.000      1.000  1.000     0.500
silent+learned-spec vs greedy+silent-centre  -12.000    1.000  1.000     0.000
silent+learned-spec vs greedy+random-centre  -4.000     1.000  1.000     0.000
silent+learned-spec vs no-comm               -12.500    1.000  1.000     0.000
silent+learned-spec vs oracle                -47.500    1.000  1.000     0.000
greedy+hand-coded vs random-actions          19.000     1.000  1.000     1.000
greedy+hand-coded vs greedy+silent-centre    7.000      1.000  1.000     1.000
greedy+hand-coded vs greedy+random-centre    15.000     1.000  1.000     1.000
greedy+hand-coded vs no-comm                 6.500      1.000  1.000     1.000
greedy+hand-coded vs oracle                  -28.500    1.000  1.000     0.000

system               changed_random_matched  changed_random_matched_ci  changed_shuffle_batch  changed_shuffle_batch_ci  cic_bits  cic_bits_ci  distinct_codes  distinct_codes_ci  js_random_matched_bits  js_random_matched_bits_ci  js_shuffle_batch_bits  js_shuffle_batch_bits_ci  message_entropy_bits  message_entropy_bits_ci  mi_bits  mi_bits_ci  mi_direction_bits  mi_direction_bits_ci  mi_far_bits  mi_far_bits_ci  mi_null_q95_bits  mi_null_q95_bits_ci  mi_p_value  mi_p_value_ci  target_entropy_bits  target_entropy_bits_ci  topsim  topsim_ci  utilization  utilization_ci
-------------------  ----------------------  -------------------------  ---------------------  ------------------------  --------  -----------  --------------  -----------------  ----------------------  -------------------------  ---------------------  ------------------------  --------------------  -----------------------  -------  ----------  -----------------  --------------------  -----------  --------------  ----------------  -------------------  ----------  -------------  -------------------  ----------------------  ------  ---------  -----------  --------------
silent+learned-spec  -                       -                          -                      -                         -         -            1.000           nan                -                       -                          -                      -                         0.000                 nan                      0.000    nan         0.000              nan                   0.000        nan             0.000             nan                  1.000       nan            2.642                nan                     nan     nan        0.000        nan           
greedy+hand-coded    0.325                   nan                        0.300                  nan                       0.755     nan          7.000           nan                0.333                   nan                        0.289                  nan                       2.491                 nan                      2.504    nan         1.843              nan                   0.725        nan             0.052             nan                  0.010       nan            2.491                nan                     0.777   nan        0.830        nan           
```
