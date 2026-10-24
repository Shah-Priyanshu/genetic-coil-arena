# Squad: learned protocol vs hand-coded protocol at 3 bits (R6)

- Date: 2026-10-23
- Commit: 5ba43b8 (code; this report is committed on top of it)
- Compute: Intel(R) Xeon(R) Processor @ 2.80GHz, 4 logical cores, 4 worker processes, numpy/numba only (no GPU). Training wall time 35.3 min (2.11 CPU-hours), evaluation 0.4 min.
- Training: 4 independent runs per condition and radius; 1000 ES generations × population 64 × 8 episodes per member (≈512k episodes per run).
- Held-out evaluation: run s plays episodes 3,500,000,000 + s × 10,000 + i, i < 100. Training runs are the independent units for every CI and test (n = number of runs), so the CIs include training-seed variance, not only episode noise.
- Squad: 21×21 walls, 3 snakes, 2 food per colour, health 60, T = 300; channel = 1 symbol, V = 8 (3 bits) per receiver, sent every step (k = 1, the hand-coded centre's interval), read one step later.

Reproduce with `python scripts/learned_protocol.py --scale cpu --radii 3 --runs 4 --generations 1000 --population 64 --episodes-per-member 8 --workers 4 --eval-episodes 100 --out docs/results/learned_protocol.md`.

## Design

- **Algorithm:** OpenAI-ES (`learning.EvolutionStrategy`: antithetic Gaussian perturbations σ = 0.05, centred-rank shaping, Adam lr = 0.03, weight decay 0.005). Centre and specialists are one parameter vector, so the joint system is optimised end-to-end from the team return through the discrete argmax channel, with no gradient through the message. All members of a generation play the same episodes and action noise. Fitness = team return (food + 0.01 per living snake per step). Same budget, sizes and hyperparameters for every condition; the three conditions of run s start from the same specialist weights.
- **Specialist (shared by the 3 snakes):** egocentric window planes (4·(2r+1)²), health, one-hot received code (8) → 16 tanh → 3 action logits, actions sampled from the softmax (3347 parameters at r = 3).
- **Learned centre (shared across receivers):** per receiver, its two own-colour food items as egocentric (ahead, right) / 21 + valid flag, sorted near to far, and its health (7 inputs) → 16 tanh → 8 logits; the code is the argmax (264 parameters). It gets food positions, not directions or distance classes: which partition of the plane to name is learned. Sorting by distance only removes a slot permutation.
- **Conditions:** (a) `silent+learned-spec`: the centre always sends code 0; (b) `hand-coded+learned-spec`: the fixed `HandCodedCentre` (`2·direction + far`); (c) `learned-centre+learned-spec`. Plus the scripted references `greedy+hand-coded` and the registered baselines (random actions, greedy + silent / random centre, no-comm greedy, full-view oracle).
- **Checkpoint:** the ES mean is scored every 10 generations on 32 fixed validation episodes (seeds 4,000,000,000 + run × 10,000 + i, disjoint from training and test); the best checkpoint is evaluated on the test seeds.
- **Metrics:** computed per training run and averaged (runs learn different codes, so messages are never pooled). T = the receiver's nearest own-colour food: direction relative to heading (4) × near/far (Manhattan > 5) + 'no food' (the hand-coded meaning). topsim = Spearman between Hamming distance of (direction, far) and of the code, on receivers with food. CIC from the specialists' exact softmax.

## Interpretation

- **r = 3:** team food (a) silent 10.28, (b) hand-coded 30.73, (c) learned 22.50; scripted greedy+hand-coded 18.72, no-comm 8.30, oracle 42.83. (c) − (a): +12.22 [-2.44, +26.87], not distinguishable, p(Holm) = 0.600. (c) − (b): -8.23 [-22.38, +5.91], not distinguishable, p(Holm) = 0.600. (b) − (a): +20.45 [+16.06, +24.84], higher, p(Holm) = 0.143. The learned centre uses 0.90 of 3 bits with MI(M;T) = 0.53 bits; CIC 0.273 bits (means over runs); interventions whose drop CI excludes 0: `zero`.

### Discussion (hand-written from the tables below)

**Verdict at r = 3: a learned protocol can reach the hand-coded protocol's performance with about one bit, but it does not beat it, and one run in four never forms a code. On this compute the evidence favours learning the listeners over learning the language.**

- **Is learned communication useful at all? Yes when a code emerges, but n = 4 cannot establish it.** In 3 of 4 runs the jointly trained squad built a code. Those runs ate 20.2, 28.1 and 31.3 food, against 8.8 to 11.1 for their own silent controls and 18.7 for the scripted greedy + hand-coded squad. Zeroing or shuffling those codes costs 15 to 30 food. In run 2 no code emerged: the centre sent one symbol almost all the time (H(M) = 0.05 bits, MI = 0.01). That squad ended at its silent control's level (10.45 vs 10.46). Because of this bimodality, (c) − (a) = +12.2 with CI [−2.4, +26.9].
- **Learned vs hand-coded protocol (for learned listeners): no better, and on average worse.** (b) = 30.7 ± 4.6 and (c) = 22.5 ± 14.8, so (c) − (b) = −8.2 [−22.4, +5.9]. The two good learned runs (28.1 and 31.3) fall inside (b)'s range (26.9 to 33.9). Run 1 beat the hand-coded run trained from the same initial specialist (31.3 vs 26.9). The average cost of learning the language is the failed run plus the slower start: the validation curves reach 13.8 vs 26.1 food at generation 500.
- **What the learned code says.** It uses 2 or 3 symbols: H(M) = 1.0 to 1.6 of the 3 bits. Nearly all of the information is direction: MI(M; direction) = 0.6 to 0.7 bits and MI(M; far) ≤ 0.05.
  - Run 0's code is a single bit: "your nearest food is ahead or to your right" vs "behind or to your left".
  - Run 1 has three symbols that roughly split right/behind, left and ahead-far.
  - No run encodes distance.
  
  So the learned centre rediscovers the direction half of the hand-coded meaning, coarsened to a few sectors, and drops near/far. Topographic similarity is lower than the hand-coded code's (0.23 vs 0.74) because the code has fewer distinctions. Dropping near/far may cost little: the scripted greedy listener also ignores the far bit and reads only `(code // 2) % 4`.
- **The larger effect is learning the listener.** Learned specialists reading the fixed hand-coded centre eat 30.7 ± 4.6. That is +12.0 [+7.5, +16.5] over the scripted greedy listener reading the same messages, and 72% of the full-view oracle (42.8). Learned specialists also use their window better: the silent learned squad (10.3) beats the no-comm greedy baseline (8.3). Learned listeners lean on the message less than greedy does (CIC 0.47 bits for (b), 0.27 for (c), against 0.86 for the scripted squad), yet they get more out of it.
- **Ablations.**
  - For learned listeners, "zero" is not silence: code 0 is either unused or has a learned meaning, so zeroing sends a wrong message. Under it (b) drops to 3.9 food, below its own silent control (10.3). For the scripted squad, zero equals the silent-centre baseline, as R5 checked.
  - The silent system's four interventions change nothing, exactly. This is the deaf control.
  - Only `zero` has a drop CI that excludes 0 for (c), because run 2 barely reacts to shuffling (drop 0.1).
- **Statistics.** Training runs are the unit, and n = 4 per condition.
  - The smallest attainable p-values are 0.125 for a two-sided Wilcoxon test (0.5 after Holm over four interventions) and 0.029 for a 4-vs-4 Mann-Whitney test (0.143 after Holm over five comparisons). So no comparison here can pass a test at 0.05.
  - Read the Welch CIs instead. (b) > (a), (b) > scripted, and every (b) ablation drop exclude zero. Every (c) comparison is inconclusive.
- **Why run 2 failed (diagnosis, no fix tried).** This is the chicken-and-egg problem of emergent communication. Early on, the specialists ignore the message, so nothing rewards the centre for informative codes. The centre drifts to a near-constant argmax, so nothing rewards the specialists for listening.
  - In run 2 the listeners did learn to rely on the constant symbol as a bias: zeroing it costs 5.4 food, shuffling costs 0.1.
  - Run 3 was still improving at generation 1000 (best checkpoint at 990), and the mean learned curve rose from 17.5 to 21.6 food between generations 750 and 1000. Part of the gap to (b) is therefore undertraining.
  - The 90-minute CPU budget went to the four mandatory r = 3 runs per condition and a reduced r = 5 run, so neither fix was tried. The well-motivated fixes are:
    1. Longer training plus several restarts per run, selected on the validation seeds.
    2. A positive-signalling term that keeps the centre's codes diverse while the listeners catch up, for example λ·H(M) per generation added to the fitness, or a softmax-sampled centre with an annealed temperature (Eccles et al. 2019).
- **r = 5** (separate, reduced-budget report: `learned_protocol_r5.md`, 500 generations). Every learned condition is still far from converged. The silent learned squad reaches only 2.9 food, against 15.7 for no-comm greedy; its 4·11² = 484-pixel window is slow to learn with ES. The ordering is the same as at r = 3: (b) 12.7 ≈ (c) 11.0 ≫ (a) 2.9. In 3 of 4 runs a 1 to 1.4-bit direction code emerged, and run 0 failed in the same way as r = 3's run 2. A message is a shortcut that learning finds before it finds foraging from the window: (b) − (a) = +9.8 [+2.0, +17.6]. No r = 5 conclusion about the learned protocol's final quality is possible at this budget.

## Team food per episode (held-out)

Mean ± 95% t-CI over training runs (each run's score is its mean over the test episodes; scripted systems and baselines use the same per-run episode sets).

| condition | role | r = 3 |
|---|---|---|
| silent+learned-spec | system | 10.28 ± 1.62 |
| hand-coded+learned-spec | system | 30.73 ± 4.61 |
| learned-centre+learned-spec | system | 22.50 ± 14.77 |
| greedy+hand-coded | system | 18.72 ± 0.90 |
| random-actions | baseline | 0.33 ± 0.13 |
| greedy+silent-centre | baseline | 6.88 ± 0.61 |
| greedy+random-centre | baseline | 3.57 ± 0.31 |
| no-comm | baseline | 8.30 ± 0.44 |
| oracle | baseline | 42.83 ± 0.46 |

## Headline comparisons

(c) vs (a): is learned communication useful at all; (c) vs (b): learned vs hand-coded protocol for learned listeners; (b) vs (a): do learned listeners use the hand-coded channel. Unpaired over training runs.

| r | comparison | mean diff [95% Welch CI] | p (Mann-Whitney, Holm) | A12 |
|---|---|---|---|---|
| 3 | learned-centre+learned-spec vs silent+learned-spec | 12.22 [-2.44, 26.87] | 0.600 | 0.81 |
| 3 | learned-centre+learned-spec vs hand-coded+learned-spec | -8.23 [-22.38, 5.91] | 0.600 | 0.19 |
| 3 | hand-coded+learned-spec vs silent+learned-spec | 20.45 [16.06, 24.84] | 0.143 | 1.00 |
| 3 | learned-centre+learned-spec vs greedy+hand-coded | 3.78 [-10.96, 18.51] | 0.600 | 0.75 |
| 3 | hand-coded+learned-spec vs greedy+hand-coded | 12.01 [7.49, 16.53] | 0.143 | 1.00 |

## Message ablations (learned systems)

Drop = intact − ablated team food, paired by training run; p = Wilcoxon signed-rank, Holm across the four interventions per system.

| r | system | intervention | intact | ablated | drop [95% CI] | p (Holm) | A12 | listening |
|---|---|---|---|---|---|---|---|---|
| 3 | silent+learned-spec | zero | 10.28 | 10.28 | 0.00 [0.00, 0.00] | 1.000 | 0.50 | no |
| 3 | silent+learned-spec | shuffle_batch | 10.28 | 10.28 | 0.00 [0.00, 0.00] | 1.000 | 0.50 | no |
| 3 | silent+learned-spec | swap_receivers | 10.28 | 10.28 | 0.00 [0.00, 0.00] | 1.000 | 0.50 | no |
| 3 | silent+learned-spec | random_matched | 10.28 | 10.28 | 0.00 [0.00, 0.00] | 1.000 | 0.50 | no |
| 3 | hand-coded+learned-spec | zero | 30.73 | 3.88 | 26.86 [23.21, 30.50] | 0.500 | 1.00 | yes |
| 3 | hand-coded+learned-spec | shuffle_batch | 30.73 | 2.09 | 28.64 [24.09, 33.19] | 0.500 | 1.00 | yes |
| 3 | hand-coded+learned-spec | swap_receivers | 30.73 | 8.03 | 22.71 [18.66, 26.76] | 0.500 | 1.00 | yes |
| 3 | hand-coded+learned-spec | random_matched | 30.73 | 2.61 | 28.13 [23.56, 32.70] | 0.500 | 1.00 | yes |
| 3 | learned-centre+learned-spec | zero | 22.50 | 2.97 | 19.52 [1.86, 37.19] | 0.500 | 1.00 | yes |
| 3 | learned-centre+learned-spec | shuffle_batch | 22.50 | 5.20 | 17.29 [-3.26, 37.85] | 0.500 | 1.00 | no |
| 3 | learned-centre+learned-spec | swap_receivers | 22.50 | 9.19 | 13.31 [-2.10, 28.72] | 0.500 | 1.00 | no |
| 3 | learned-centre+learned-spec | random_matched | 22.50 | 5.53 | 16.97 [-3.28, 37.23] | 0.500 | 0.94 | no |

## What the centre says and how much it moves actions

Mean ± 95% CI over training runs; MI null q95 from 100 target shuffles (mean over runs).

| r | squad | H(M) bits | used / 3 bits | codes used | H(T) bits | MI(M;T) bits | MI null q95 | MI(M;direction) | MI(M;far) | topsim | CIC bits |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | silent+learned-spec | 0.000 ± 0.000 | 0% | 1.0 ± 0.0 | 2.485 ± 0.080 | 0.000 ± 0.000 | 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | n/a | n/a |
| 3 | hand-coded+learned-spec | 2.677 ± 0.133 | 89% | 8.0 ± 0.0 | 2.677 ± 0.133 | 2.677 ± 0.133 | 0.001 | 1.750 ± 0.084 | 0.936 ± 0.039 | 0.737 ± 0.026 | 0.466 ± 0.117 |
| 3 | learned-centre+learned-spec | 0.899 ± 1.005 | 30% | 2.8 ± 0.8 | 2.674 ± 0.182 | 0.528 ± 0.566 | 0.000 | 0.495 ± 0.521 | 0.016 ± 0.037 | 0.232 ± 0.229 | 0.273 ± 0.291 |
| 3 | greedy+hand-coded | 2.634 ± 0.008 | 88% | 8.0 ± 0.0 | 2.634 ± 0.008 | 2.635 ± 0.008 | 0.001 | 1.910 ± 0.013 | 0.769 ± 0.013 | 0.748 ± 0.006 | 0.860 ± 0.016 |

| r | squad | JS(shuffle_batch) bits | action changed (shuffle_batch) | JS(random_matched) bits | action changed (random_matched) |
|---|---|---|---|---|---|
| 3 | hand-coded+learned-spec | 0.170 ± 0.048 | 0.218 ± 0.063 | 0.172 ± 0.047 | 0.221 ± 0.063 |
| 3 | learned-centre+learned-spec | 0.134 ± 0.126 | 0.194 ± 0.178 | 0.128 ± 0.123 | 0.186 ± 0.173 |
| 3 | greedy+hand-coded | 0.396 ± 0.010 | 0.410 ± 0.020 | 0.404 ± 0.023 | 0.417 ± 0.022 |

### Learned centre, run by run

(a), (b), (c) = silent, hand-coded and learned centre with learned specialists trained from the same initial specialist weights.

| r | run | team food (c) | (b) | (a) | H(M) bits | codes used | MI(M;T) | MI(M;direction) | MI(M;far) | CIC bits | drop when zeroed | drop when shuffled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0 | 28.09 | 31.39 | 8.81 | 0.98 | 2 | 0.67 | 0.66 | 0.00 | 0.334 | 26.45 | 26.41 |
| 3 | 1 | 31.25 | 26.87 | 10.74 | 1.57 | 3 | 0.81 | 0.71 | 0.05 | 0.411 | 30.08 | 28.03 |
| 3 | 2 | 10.45 | 30.80 | 10.46 | 0.05 | 3 | 0.01 | 0.01 | 0.00 | 0.003 | 5.42 | 0.11 |
| 3 | 3 | 20.20 | 33.87 | 11.12 | 1.00 | 3 | 0.62 | 0.60 | 0.01 | 0.343 | 16.15 | 14.63 |

### Learned codebooks

r = 3, training run 0 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3638 | 5105 | 567 | 1025 | 7 | 0 | 229 | 976 | 0 |
| 7 | 60 | 0 | 96 | 214 | 653 | 1937 | 1734 | 3415 | 0 |

r = 3, training run 1 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1440 | 705 | 1175 | 2758 | 483 | 1174 | 12 | 0 | 0 |
| 6 | 950 | 143 | 5 | 0 | 232 | 505 | 1854 | 1906 | 0 |
| 7 | 1368 | 4538 | 0 | 0 | 0 | 0 | 159 | 845 | 0 |

r = 3, training run 2 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 0 | 2 | 0 | 0 | 1 | 31 | 2 | 13 | 0 |
| 5 | 0 | 0 | 0 | 0 | 0 | 12 | 0 | 0 | 0 |
| 7 | 1428 | 3106 | 584 | 4525 | 282 | 1910 | 432 | 844 | 0 |

r = 3, training run 3 (counts of live receivers; rows = code sent, columns = the receiver's nearest own-colour food):

| code | ahead/near | ahead/far | right/near | right/far | behind/near | behind/far | left/near | left/far | no food |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 1994 | 3842 | 211 | 191 | 74 | 50 | 887 | 1154 | 0 |
| 3 | 485 | 245 | 1505 | 4674 | 472 | 1847 | 127 | 57 | 0 |
| 6 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |

## Learning curves

Full curves (every 10 generations, every run) in `learned_protocol_curves.csv`.

r = 3 (validation team food of the ES mean, mean over runs):

| condition | gen 0 | gen 100 | gen 250 | gen 500 | gen 750 | gen 1000 | best ckpt gen (mean) |
|---|---|---|---|---|---|---|---|
| silent+learned-spec | 0.37 | 1.36 | 3.10 | 6.28 | 9.07 | 9.71 | 922 |
| hand-coded+learned-spec | 0.30 | 3.69 | 17.82 | 26.14 | 30.43 | 31.78 | 858 |
| learned-centre+learned-spec | 0.32 | 1.39 | 6.84 | 13.84 | 17.46 | 21.59 | 910 |

## Harness output

`evaluation.format_table` per radius.

### r = 3

```
Task: squad-learned-protocol-r3  (seeds: 4)
Baselines: random-actions, greedy+silent-centre, greedy+random-centre, no-comm, oracle

condition                    role      n  mean ± 95% CI    IQM [95% CI]           
---------------------------  --------  -  ---------------  -----------------------
silent+learned-spec          system    4  10.283 ± 1.620   10.600 [8.810, 11.120] 
hand-coded+learned-spec      system    4  30.733 ± 4.612   31.095 [26.870, 33.870]
learned-centre+learned-spec  system    4  22.498 ± 14.765  24.145 [10.450, 31.250]
greedy+hand-coded            system    4  18.723 ± 0.898   18.965 [17.880, 19.080]
random-actions               baseline  4  0.335 ± 0.133    0.295 [0.290, 0.460]   
greedy+silent-centre         baseline  4  6.880 ± 0.607    6.735 [6.610, 7.440]   
greedy+random-centre         baseline  4  3.570 ± 0.312    3.495 [3.430, 3.860]   
no-comm                      baseline  4  8.305 ± 0.438    8.320 [8.000, 8.580]   
oracle                       baseline  4  42.828 ± 0.464   42.905 [42.430, 43.070]

system                       intervention    ablated mean  drop [95% CI]            p (Holm)  A12    listening
---------------------------  --------------  ------------  -----------------------  --------  -----  ---------
silent+learned-spec          zero            10.283        0.000 [0.000, 0.000]     1.000     0.500  no       
silent+learned-spec          shuffle_batch   10.283        0.000 [0.000, 0.000]     1.000     0.500  no       
silent+learned-spec          swap_receivers  10.283        0.000 [0.000, 0.000]     1.000     0.500  no       
silent+learned-spec          random_matched  10.283        0.000 [0.000, 0.000]     1.000     0.500  no       
hand-coded+learned-spec      zero            3.875         26.857 [23.212, 30.503]  0.500     1.000  yes      
hand-coded+learned-spec      shuffle_batch   2.093         28.640 [24.092, 33.188]  0.500     1.000  yes      
hand-coded+learned-spec      swap_receivers  8.025         22.707 [18.657, 26.758]  0.500     1.000  yes      
hand-coded+learned-spec      random_matched  2.605         28.127 [23.559, 32.696]  0.500     1.000  yes      
learned-centre+learned-spec  zero            2.973         19.525 [1.861, 37.189]   0.500     1.000  yes      
learned-centre+learned-spec  shuffle_batch   5.203         17.295 [-3.263, 37.853]  0.500     1.000  no       
learned-centre+learned-spec  swap_receivers  9.188         13.310 [-2.104, 28.724]  0.500     1.000  no       
learned-centre+learned-spec  random_matched  5.525         16.972 [-3.281, 37.226]  0.500     0.938  no       

comparison                                           mean diff  p      p (Holm)  A12  
---------------------------------------------------  ---------  -----  --------  -----
silent+learned-spec vs random-actions                9.948      0.029  0.571     1.000
silent+learned-spec vs greedy+silent-centre          3.402      0.029  0.571     1.000
silent+learned-spec vs greedy+random-centre          6.713      0.029  0.571     1.000
silent+learned-spec vs no-comm                       1.978      0.029  0.571     1.000
silent+learned-spec vs oracle                        -32.545    0.029  0.571     0.000
hand-coded+learned-spec vs random-actions            30.398     0.029  0.571     1.000
hand-coded+learned-spec vs greedy+silent-centre      23.852     0.029  0.571     1.000
hand-coded+learned-spec vs greedy+random-centre      27.163     0.029  0.571     1.000
hand-coded+learned-spec vs no-comm                   22.428     0.029  0.571     1.000
hand-coded+learned-spec vs oracle                    -12.095    0.029  0.571     0.000
learned-centre+learned-spec vs random-actions        22.163     0.029  0.571     1.000
learned-centre+learned-spec vs greedy+silent-centre  15.618     0.029  0.571     1.000
learned-centre+learned-spec vs greedy+random-centre  18.928     0.029  0.571     1.000
learned-centre+learned-spec vs no-comm               14.193     0.029  0.571     1.000
learned-centre+learned-spec vs oracle                -20.330    0.029  0.571     0.000
greedy+hand-coded vs random-actions                  18.387     0.029  0.571     1.000
greedy+hand-coded vs greedy+silent-centre            11.842     0.029  0.571     1.000
greedy+hand-coded vs greedy+random-centre            15.152     0.029  0.571     1.000
greedy+hand-coded vs no-comm                         10.418     0.029  0.571     1.000
greedy+hand-coded vs oracle                          -24.105    0.029  0.571     0.000

system                       changed_random_matched  changed_random_matched_ci  changed_shuffle_batch  changed_shuffle_batch_ci  cic_bits  cic_bits_ci  distinct_codes  distinct_codes_ci  js_random_matched_bits  js_random_matched_bits_ci  js_shuffle_batch_bits  js_shuffle_batch_bits_ci  message_entropy_bits  message_entropy_bits_ci  mi_bits  mi_bits_ci  mi_direction_bits  mi_direction_bits_ci  mi_far_bits  mi_far_bits_ci  mi_null_q95_bits  mi_null_q95_bits_ci  mi_p_value  mi_p_value_ci  target_entropy_bits  target_entropy_bits_ci  topsim  topsim_ci  utilization  utilization_ci
---------------------------  ----------------------  -------------------------  ---------------------  ------------------------  --------  -----------  --------------  -----------------  ----------------------  -------------------------  ---------------------  ------------------------  --------------------  -----------------------  -------  ----------  -----------------  --------------------  -----------  --------------  ----------------  -------------------  ----------  -------------  -------------------  ----------------------  ------  ---------  -----------  --------------
silent+learned-spec          -                       -                          -                      -                         -         -            1.000           0.000              -                       -                          -                      -                         0.000                 0.000                    0.000    0.000       0.000              0.000                 0.000        0.000           0.000             0.000                1.000       0.000          2.485                0.080                   nan     nan        0.000        0.000         
hand-coded+learned-spec      0.221                   0.063                      0.218                  0.063                     0.466     0.117        8.000           0.000              0.172                   0.047                      0.170                  0.048                     2.677                 0.133                    2.677    0.133       1.750              0.084                 0.936        0.039           0.001             0.000                0.010       0.000          2.677                0.133                   0.737   0.026      0.892        0.044         
learned-centre+learned-spec  0.186                   0.173                      0.194                  0.178                     0.273     0.291        2.750           0.796              0.128                   0.123                      0.134                  0.126                     0.899                 1.005                    0.528    0.566       0.495              0.521                 0.016        0.037           0.000             0.000                0.010       0.000          2.674                0.182                   0.232   0.229      0.300        0.335         
greedy+hand-coded            0.417                   0.022                      0.410                  0.020                     0.860     0.016        8.000           0.000              0.404                   0.023                      0.396                  0.010                     2.634                 0.008                    2.635    0.008       1.910              0.013                 0.769        0.013           0.001             0.000                0.010       0.000          2.634                0.008                   0.748   0.006      0.878        0.003         
```
