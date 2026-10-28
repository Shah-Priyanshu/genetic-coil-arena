# R7b: a learned command centre that follows a GoalSpec (goal G7)

> **16-run report** (2026-10-28): seeds 0–7 from jobs `r7b3-train-*` and seeds 8–15 from `r7b4-train-*` and `r7b5-train-blind`, all at 10k generations. Built locally with `--from-json "<gpu-results>/runs/r7b[345]-train-*/artifacts"`; the reproduce line at the end does not match these job names.
> - **G7 still passes, through (c).** With the goal at the centre and the specialists, (c) reaches 92% of the hand-coded centre (d) on held-out goals, 21.31 against 23.07. With 16 runs, (c) − (d) = −1.76 ± 1.71 is now just significant (p = 0.048).
> - **(b) falls to 85% of (d)** (19.68; (b) − (d) = −3.39 ± 2.95, p = 0.010) because 2 of its 16 runs failed to train. Seeds 10 and 15 score 6.6 and 8.0 on held-out goals, against 18.7 to 23.7 for the other 14, which average 21.46 (93% of (d)). IQM, which trims them, gives 21.21 [18.93, 22.25].
> - **The failures show on the training goals.** Those two runs' best validation scores were already 5.7 and 8.2, against 18.1 to 22.4 for the rest, so they could be caught without held-out data. No such rule was pre-registered, and none is applied here.
> - **(c) had no failed run:** all 16 score 18.6 to 25.7. The hand-coded centre (d) has two weaker runs (17.1, 17.9).
> - **Reading:** a goal carried only by the 3-bit messages is learnable in most runs but less reliable to train. Giving the goal to the specialists as well costs nothing on held-out goals and removed the failures.

**G7 verdict: PASS.** a goal-conditioned centre beats (a) and reaches 90% of (d) on held-out goals.

Scale: full. 16 training runs per condition; every number is a mean over training runs with a 95% CI (Student t unless stated). Mission score = eater food - 1.0 x non-eater food - death penalty (0 / 2 / 5 for low / normal / high caution) x dead eaters (see `experiments/r7b_goal_conditioned.py`). 15 training goals, 6 held-out goals: eater subset {green, blue} at every caution (`novel_subset`) and three seen-part combinations (`novel_combo`: red/high, red+blue/normal, red+green+blue/low).

Training settings: `{"radius": 3, "generations": 10000, "population": 64, "reps": 1, "sigma": 0.05, "lr": 0.03, "weight_decay": 0.005, "spec_hidden": 24, "centre_hidden": 24, "eval_every": 50, "val_reps": 2, "max_steps": 300}`.
Compute: training 708.4 CPU-hours (sum over runs); evaluation 0.9 min.

## Mission score

| Condition | Seen goals | Held-out goals | Novel subset | Novel combo |
|---|---|---|---|---|
| (a) blind-centre+learned-spec | -3.99 ± 0.05 | -3.35 ± 0.14 | -4.20 ± 0.12 | -2.51 ± 0.17 |
| (b) goal-centre+learned-spec | 16.40 ± 2.70 | 19.68 ± 2.68 | 19.56 ± 2.83 | 19.81 ± 2.64 |
| (c) goal-centre+goal-spec | 18.32 ± 0.84 | 21.31 ± 1.07 | 21.24 ± 1.13 | 21.38 ± 1.23 |
| (d) hand-goal-centre+learned-spec | 18.97 ± 1.28 | 23.07 ± 1.42 | 22.76 ± 1.66 | 23.38 ± 1.28 |
| random-actions | -4.04 ± 0.02 | -3.73 ± 0.03 | -4.54 ± 0.04 | -2.91 ± 0.04 |
| greedy+hand-coded (goal-blind) | -1.87 ± 0.62 | 3.50 ± 0.54 | 2.78 ± 0.67 | 4.22 ± 0.61 |
| (e) scripted-wrapper (R7a) | -0.18 ± 0.14 | 0.87 ± 0.18 | 0.09 ± 0.21 | 1.66 ± 0.29 |

## G7

Pass: (b) or (c) beats (a) on held-out goals (Welch CI of the difference above 0, Holm-corrected Mann-Whitney p < 0.05) and reaches at least 90% of (d). Kill: neither (b) nor (c) beats (a).

(a) = -3.35, (d) = 23.07 on held-out goals.

| Condition | Held-out mean | Diff vs (a) | Welch CI | p (Holm) | Beats (a) | % of (d) |
|---|---|---|---|---|---|---|
| (b) goal-centre+learned-spec | 19.68 | +23.04 | ± 2.69 | 3.09e-06 | yes | 85% |
| (c) goal-centre+goal-spec | 21.31 | +24.66 | ± 1.07 | 3.09e-06 | yes | 92% |

### Pairwise differences, heldout goals

| A | B | A - B | Welch CI | p (Holm) |
|---|---|---|---|---|
| (b) goal-centre+learned-spec | (a) blind-centre+learned-spec | +23.04 | ± 2.69 | 1.23e-05 |
| (c) goal-centre+goal-spec | (a) blind-centre+learned-spec | +24.66 | ± 1.07 | 1.23e-05 |
| (d) hand-goal-centre+learned-spec | (a) blind-centre+learned-spec | +26.42 | ± 1.43 | 1.23e-05 |
| (b) goal-centre+learned-spec | (d) hand-goal-centre+learned-spec | -3.39 | ± 2.95 | 0.00985 |
| (c) goal-centre+goal-spec | (d) hand-goal-centre+learned-spec | -1.76 | ± 1.71 | 0.0475 |
| (c) goal-centre+goal-spec | (b) goal-centre+learned-spec | +1.62 | ± 2.83 | 1 |
| (e) scripted-wrapper (R7a) | (a) blind-centre+learned-spec | +4.23 | ± 0.22 | 1.23e-05 |
| (d) hand-goal-centre+learned-spec | (e) scripted-wrapper (R7a) | +22.20 | ± 1.43 | 1.23e-05 |

### Pairwise differences, seen goals

| A | B | A - B | Welch CI | p (Holm) |
|---|---|---|---|---|
| (b) goal-centre+learned-spec | (a) blind-centre+learned-spec | +20.39 | ± 2.70 | 1.22e-05 |
| (c) goal-centre+goal-spec | (a) blind-centre+learned-spec | +22.31 | ± 0.84 | 1.22e-05 |
| (d) hand-goal-centre+learned-spec | (a) blind-centre+learned-spec | +22.96 | ± 1.28 | 1.22e-05 |
| (b) goal-centre+learned-spec | (d) hand-goal-centre+learned-spec | -2.57 | ± 2.91 | 0.0997 |
| (c) goal-centre+goal-spec | (d) hand-goal-centre+learned-spec | -0.65 | ± 1.47 | 0.326 |
| (c) goal-centre+goal-spec | (b) goal-centre+learned-spec | +1.92 | ± 2.79 | 0.51 |
| (e) scripted-wrapper (R7a) | (a) blind-centre+learned-spec | +3.81 | ± 0.15 | 1.22e-05 |
| (d) hand-goal-centre+learned-spec | (e) scripted-wrapper (R7a) | +19.15 | ± 1.28 | 1.22e-05 |

## Do the messages carry the goal?

### heldout goals (live receivers; MI in bits, Miller-Madow; null = 95th percentile of shuffled labels; probe = held-out-episode accuracy vs majority class)

| Condition | H(msg) bits | MI own-eater (null) | probe own-eater (chance) | MI caution (null) | probe caution (chance) | MI goal (null) |
|---|---|---|---|---|---|---|
| (a) blind-centre+learned-spec | 1.16 ± 0.31 | 0.000 ± 0.000 (0.001 ± 0.000) | 0.671 ± 0.008 (0.671 ± 0.008) | 0.001 ± 0.001 (0.001 ± 0.000) | 0.328 ± 0.008 (0.321 ± 0.009) | 0.002 ± 0.001 (0.001 ± 0.000) |
| (b) goal-centre+learned-spec | 0.97 ± 0.16 | 0.147 ± 0.042 (0.000 ± 0.000) | 0.921 ± 0.027 (0.905 ± 0.016) | 0.013 ± 0.007 (0.000 ± 0.000) | 0.339 ± 0.031 (0.322 ± 0.038) | 0.026 ± 0.010 (0.000 ± 0.000) |
| (c) goal-centre+goal-spec | 0.99 ± 0.03 | 0.043 ± 0.025 (0.000 ± 0.000) | 0.904 ± 0.014 (0.903 ± 0.014) | 0.012 ± 0.009 (0.000 ± 0.000) | 0.313 ± 0.033 (0.359 ± 0.032) | 0.021 ± 0.014 (0.000 ± 0.000) |
| (d) hand-goal-centre+learned-spec | 2.47 ± 0.03 | 0.523 ± 0.011 (0.000 ± 0.000) | 1.000 ± 0.000 (0.882 ± 0.004) | 0.025 ± 0.002 (0.000 ± 0.000) | 0.340 ± 0.038 (0.352 ± 0.030) | 0.060 ± 0.003 (0.000 ± 0.000) |

### seen goals (live receivers; MI in bits, Miller-Madow; null = 95th percentile of shuffled labels; probe = held-out-episode accuracy vs majority class)

| Condition | H(msg) bits | MI own-eater (null) | probe own-eater (chance) | MI caution (null) | probe caution (chance) | MI goal (null) |
|---|---|---|---|---|---|---|
| (a) blind-centre+learned-spec | 1.16 ± 0.31 | 0.000 ± 0.000 (0.000 ± 0.000) | 0.518 ± 0.017 (0.528 ± 0.014) | 0.001 ± 0.000 (0.001 ± 0.000) | 0.336 ± 0.008 (0.338 ± 0.009) | 0.005 ± 0.002 (0.003 ± 0.000) |
| (b) goal-centre+learned-spec | 1.02 ± 0.15 | 0.228 ± 0.062 (0.000 ± 0.000) | 0.882 ± 0.041 (0.856 ± 0.023) | 0.006 ± 0.003 (0.000 ± 0.000) | 0.355 ± 0.008 (0.351 ± 0.009) | 0.036 ± 0.012 (0.001 ± 0.000) |
| (c) goal-centre+goal-spec | 1.01 ± 0.03 | 0.059 ± 0.035 (0.000 ± 0.000) | 0.858 ± 0.019 (0.855 ± 0.019) | 0.007 ± 0.004 (0.000 ± 0.000) | 0.353 ± 0.007 (0.350 ± 0.008) | 0.020 ± 0.008 (0.001 ± 0.000) |
| (d) hand-goal-centre+learned-spec | 2.47 ± 0.03 | 0.691 ± 0.010 (0.000 ± 0.000) | 1.000 ± 0.000 (0.817 ± 0.005) | 0.004 ± 0.001 (0.000 ± 0.000) | 0.360 ± 0.009 (0.341 ± 0.010) | 0.095 ± 0.003 (0.001 ± 0.000) |

## Ablations (score drop, intact minus intervened, paired over training runs)

Message interventions are the harness's; `goals swapped` tells the policies the goal of another episode (a different eater set) while the score uses the true goal.

### heldout goals

| Condition | random_matched | shuffle_batch | swap_receivers | zero | goals swapped |
|---|---|---|---|---|---|
| (a) blind-centre+learned-spec | +0.17 ± 0.13 | +0.19 ± 0.11 | +0.15 ± 0.11 | +0.21 ± 0.14 | +0.00 ± 0.00 |
| (b) goal-centre+learned-spec | +22.64 ± 2.93 | +22.79 ± 2.91 | +19.93 ± 2.46 | +22.86 ± 2.68 | +23.89 ± 2.69 |
| (c) goal-centre+goal-spec | +24.27 ± 0.96 | +24.30 ± 0.96 | +18.78 ± 0.88 | +23.76 ± 1.33 | +25.79 ± 1.58 |
| (d) hand-goal-centre+learned-spec | +26.24 ± 1.37 | +26.37 ± 1.38 | +22.53 ± 1.31 | +26.73 ± 1.40 | +28.07 ± 1.85 |

### seen goals

| Condition | random_matched | shuffle_batch | swap_receivers | zero | goals swapped |
|---|---|---|---|---|---|
| (a) blind-centre+learned-spec | +0.05 ± 0.08 | -0.02 ± 0.04 | +0.04 ± 0.06 | +0.02 ± 0.08 | +0.00 ± 0.00 |
| (b) goal-centre+learned-spec | +20.15 ± 2.71 | +20.29 ± 2.67 | +17.63 ± 2.41 | +20.49 ± 2.66 | +20.86 ± 2.86 |
| (c) goal-centre+goal-spec | +21.76 ± 0.69 | +21.92 ± 0.77 | +16.45 ± 1.38 | +21.60 ± 1.17 | +22.96 ± 0.82 |
| (d) hand-goal-centre+learned-spec | +22.80 ± 1.25 | +22.89 ± 1.28 | +19.91 ± 1.15 | +23.04 ± 1.27 | +23.49 ± 1.42 |

## Per-goal mean mission score (mean over training runs)

### seen

| Goal | (a) blind-centre+learned-spec | (b) goal-centre+learned-spec | (c) goal-centre+goal-spec | (d) hand-goal-centre+learned-spec | random-actions | greedy+hand-coded (goal-blind) | (e) scripted-wrapper (R7a) |
|---|---|---|---|---|---|---|---|
| r/low | -0.5 | 13.1 | 14.6 | 16.2 | -0.2 | -4.6 | 2.2 |
| r/normal | -2.6 | 12.9 | 14.1 | 14.6 | -2.1 | -7.3 | 0.8 |
| g/low | -0.7 | 13.3 | 14.8 | 15.4 | -0.1 | -5.4 | 2.3 |
| g/normal | -2.5 | 12.5 | 14.7 | 14.9 | -2.1 | -8.8 | 0.7 |
| g/high | -5.4 | 11.9 | 14.0 | 14.0 | -5.1 | -9.0 | -3.1 |
| r+g/low | 0.4 | 22.1 | 23.7 | 25.3 | 0.1 | 7.2 | 4.0 |
| r+g/normal | -3.4 | 20.1 | 22.9 | 23.1 | -3.9 | 2.5 | 1.7 |
| r+g/high | -9.6 | 17.0 | 19.9 | 20.9 | -9.9 | -1.9 | -5.3 |
| b/low | -0.5 | 13.9 | 15.8 | 15.1 | -0.0 | -7.9 | 2.0 |
| b/normal | -2.4 | 13.8 | 14.4 | 15.1 | -2.1 | -7.6 | 0.4 |
| b/high | -5.5 | 12.0 | 13.1 | 14.5 | -5.0 | -10.2 | -2.7 |
| r+b/low | 0.3 | 23.0 | 25.4 | 24.4 | 0.1 | 8.1 | 4.9 |
| r+b/high | -9.6 | 17.3 | 18.8 | 19.1 | -9.9 | -2.5 | -5.4 |
| r+g+b/normal | -4.4 | 23.9 | 27.3 | 29.1 | -5.6 | 14.0 | 2.5 |
| r+g+b/high | -13.5 | 19.1 | 21.3 | 22.9 | -14.8 | 5.4 | -7.8 |

### heldout

| Goal | (a) blind-centre+learned-spec | (b) goal-centre+learned-spec | (c) goal-centre+goal-spec | (d) hand-goal-centre+learned-spec | random-actions | greedy+hand-coded (goal-blind) | (e) scripted-wrapper (R7a) |
|---|---|---|---|---|---|---|---|
| g+b/low | 0.4 | 22.3 | 23.7 | 25.6 | 0.1 | 6.1 | 4.4 |
| g+b/normal | -3.5 | 19.9 | 21.8 | 23.3 | -3.9 | 3.0 | 1.4 |
| g+b/high | -9.5 | 16.5 | 18.2 | 19.4 | -9.9 | -0.8 | -5.6 |
| r/high | -5.4 | 10.9 | 12.5 | 14.6 | -5.1 | -9.2 | -3.1 |
| r+b/normal | -3.5 | 19.8 | 21.3 | 23.1 | -3.9 | 3.4 | 1.7 |
| r+g+b/low | 1.3 | 28.7 | 30.4 | 32.5 | 0.3 | 18.4 | 6.4 |

## Harness table, heldout goals

```
Task: squad-goal-heldout  (seeds: 16)
Baselines: random-actions, greedy+hand-coded (goal-blind), (e) scripted-wrapper (R7a)

condition                          role      n   mean ± 95% CI   IQM [95% CI]           
---------------------------------  --------  --  --------------  -----------------------
(a) blind-centre+learned-spec      system    16  -3.354 ± 0.136  -3.414 [-3.504, -3.274]
(b) goal-centre+learned-spec       system    16  19.685 ± 2.683  21.208 [18.930, 22.253]
(c) goal-centre+goal-spec          system    16  21.307 ± 1.066  20.925 [20.014, 22.266]
(d) hand-goal-centre+learned-spec  system    16  23.070 ± 1.422  23.822 [22.142, 24.576]
random-actions                     baseline  16  -3.726 ± 0.029  -3.731 [-3.759, -3.694]
greedy+hand-coded (goal-blind)     baseline  16  3.498 ± 0.541   3.397 [2.894, 3.915]   
(e) scripted-wrapper (R7a)         baseline  16  0.872 ± 0.185   0.866 [0.714, 1.032]   

system                             intervention    ablated mean  drop [95% CI]            p (Holm)  A12    listening
---------------------------------  --------------  ------------  -----------------------  --------  -----  ---------
(a) blind-centre+learned-spec      zero            -3.560        0.206 [0.068, 0.343]     0.004     0.783  yes      
(a) blind-centre+learned-spec      shuffle_batch   -3.540        0.185 [0.071, 0.300]     0.004     0.758  yes      
(a) blind-centre+learned-spec      swap_receivers  -3.504        0.149 [0.041, 0.258]     0.004     0.689  yes      
(a) blind-centre+learned-spec      random_matched  -3.523        0.169 [0.044, 0.294]     0.002     0.725  yes      
(b) goal-centre+learned-spec       zero            -3.173        22.858 [20.175, 25.541]  0.000     1.000  yes      
(b) goal-centre+learned-spec       shuffle_batch   -3.106        22.791 [19.883, 25.699]  0.000     1.000  yes      
(b) goal-centre+learned-spec       swap_receivers  -0.246        19.931 [17.474, 22.388]  0.000     1.000  yes      
(b) goal-centre+learned-spec       random_matched  -2.957        22.642 [19.709, 25.576]  0.000     1.000  yes      
(c) goal-centre+goal-spec          zero            -2.456        23.764 [22.430, 25.097]  0.000     1.000  yes      
(c) goal-centre+goal-spec          shuffle_batch   -2.993        24.300 [23.342, 25.258]  0.000     1.000  yes      
(c) goal-centre+goal-spec          swap_receivers  2.524         18.783 [17.907, 19.660]  0.000     1.000  yes      
(c) goal-centre+goal-spec          random_matched  -2.961        24.268 [23.306, 25.230]  0.000     1.000  yes      
(d) hand-goal-centre+learned-spec  zero            -3.658        26.728 [25.325, 28.131]  0.000     1.000  yes      
(d) hand-goal-centre+learned-spec  shuffle_batch   -3.296        26.366 [24.982, 27.750]  0.000     1.000  yes      
(d) hand-goal-centre+learned-spec  swap_receivers  0.544         22.526 [21.219, 23.833]  0.000     1.000  yes      
(d) hand-goal-centre+learned-spec  random_matched  -3.168        26.238 [24.867, 27.609]  0.000     1.000  yes      

comparison                                                           mean diff  p      p (Holm)  A12  
-------------------------------------------------------------------  ---------  -----  --------  -----
(a) blind-centre+learned-spec vs random-actions                      0.372      0.000  0.000     0.998
(a) blind-centre+learned-spec vs greedy+hand-coded (goal-blind)      -6.852     0.000  0.000     0.000
(a) blind-centre+learned-spec vs (e) scripted-wrapper (R7a)          -4.226     0.000  0.000     0.000
(b) goal-centre+learned-spec vs random-actions                       23.411     0.000  0.000     1.000
(b) goal-centre+learned-spec vs greedy+hand-coded (goal-blind)       16.187     0.000  0.000     1.000
(b) goal-centre+learned-spec vs (e) scripted-wrapper (R7a)           18.813     0.000  0.000     1.000
(c) goal-centre+goal-spec vs random-actions                          25.033     0.000  0.000     1.000
(c) goal-centre+goal-spec vs greedy+hand-coded (goal-blind)          17.809     0.000  0.000     1.000
(c) goal-centre+goal-spec vs (e) scripted-wrapper (R7a)              20.435     0.000  0.000     1.000
(d) hand-goal-centre+learned-spec vs random-actions                  26.796     0.000  0.000     1.000
(d) hand-goal-centre+learned-spec vs greedy+hand-coded (goal-blind)  19.572     0.000  0.000     1.000
(d) hand-goal-centre+learned-spec vs (e) scripted-wrapper (R7a)      22.198     0.000  0.000     1.000
```

## Harness table, seen goals

```
Task: squad-goal-seen  (seeds: 16)
Baselines: random-actions, greedy+hand-coded (goal-blind), (e) scripted-wrapper (R7a)

condition                          role      n   mean ± 95% CI   IQM [95% CI]           
---------------------------------  --------  --  --------------  -----------------------
(a) blind-centre+learned-spec      system    16  -3.993 ± 0.052  -3.990 [-4.032, -3.939]
(b) goal-centre+learned-spec       system    16  16.402 ± 2.699  17.824 [15.704, 18.752]
(c) goal-centre+goal-spec          system    16  18.320 ± 0.843  18.299 [17.346, 19.225]
(d) hand-goal-centre+learned-spec  system    16  18.969 ± 1.275  19.523 [18.161, 20.397]
random-actions                     baseline  16  -4.045 ± 0.023  -4.053 [-4.069, -4.032]
greedy+hand-coded (goal-blind)     baseline  16  -1.870 ± 0.619  -1.835 [-2.551, -1.105]
(e) scripted-wrapper (R7a)         baseline  16  -0.185 ± 0.140  -0.175 [-0.283, -0.054]

system                             intervention    ablated mean  drop [95% CI]            p (Holm)  A12    listening
---------------------------------  --------------  ------------  -----------------------  --------  -----  ---------
(a) blind-centre+learned-spec      zero            -4.011        0.018 [-0.058, 0.094]    0.991     0.611  no       
(a) blind-centre+learned-spec      shuffle_batch   -3.973        -0.020 [-0.061, 0.022]   0.991     0.410  no       
(a) blind-centre+learned-spec      swap_receivers  -4.035        0.042 [-0.016, 0.101]    0.391     0.625  no       
(a) blind-centre+learned-spec      random_matched  -4.042        0.049 [-0.028, 0.127]    0.391     0.650  no       
(b) goal-centre+learned-spec       zero            -4.083        20.485 [17.823, 23.148]  0.000     1.000  yes      
(b) goal-centre+learned-spec       shuffle_batch   -3.886        20.288 [17.615, 22.961]  0.000     1.000  yes      
(b) goal-centre+learned-spec       swap_receivers  -1.231        17.633 [15.226, 20.041]  0.000     1.000  yes      
(b) goal-centre+learned-spec       random_matched  -3.748        20.151 [17.436, 22.865]  0.000     1.000  yes      
(c) goal-centre+goal-spec          zero            -3.284        21.604 [20.438, 22.770]  0.000     1.000  yes      
(c) goal-centre+goal-spec          shuffle_batch   -3.598        21.918 [21.150, 22.686]  0.000     1.000  yes      
(c) goal-centre+goal-spec          swap_receivers  1.870         16.449 [15.065, 17.833]  0.000     1.000  yes      
(c) goal-centre+goal-spec          random_matched  -3.442        21.762 [21.067, 22.457]  0.000     1.000  yes      
(d) hand-goal-centre+learned-spec  zero            -4.068        23.037 [21.764, 24.310]  0.000     1.000  yes      
(d) hand-goal-centre+learned-spec  shuffle_batch   -3.919        22.888 [21.611, 24.166]  0.000     1.000  yes      
(d) hand-goal-centre+learned-spec  swap_receivers  -0.944        19.913 [18.763, 21.062]  0.000     1.000  yes      
(d) hand-goal-centre+learned-spec  random_matched  -3.836        22.805 [21.550, 24.060]  0.000     1.000  yes      

comparison                                                           mean diff  p      p (Holm)  A12  
-------------------------------------------------------------------  ---------  -----  --------  -----
(a) blind-centre+learned-spec vs random-actions                      0.052      0.019  0.019     0.744
(a) blind-centre+learned-spec vs greedy+hand-coded (goal-blind)      -2.122     0.000  0.000     0.000
(a) blind-centre+learned-spec vs (e) scripted-wrapper (R7a)          -3.808     0.000  0.000     0.000
(b) goal-centre+learned-spec vs random-actions                       20.447     0.000  0.000     1.000
(b) goal-centre+learned-spec vs greedy+hand-coded (goal-blind)       18.272     0.000  0.000     1.000
(b) goal-centre+learned-spec vs (e) scripted-wrapper (R7a)           16.587     0.000  0.000     1.000
(c) goal-centre+goal-spec vs random-actions                          22.365     0.000  0.000     1.000
(c) goal-centre+goal-spec vs greedy+hand-coded (goal-blind)          20.190     0.000  0.000     1.000
(c) goal-centre+goal-spec vs (e) scripted-wrapper (R7a)              18.505     0.000  0.000     1.000
(d) hand-goal-centre+learned-spec vs random-actions                  23.014     0.000  0.000     1.000
(d) hand-goal-centre+learned-spec vs greedy+hand-coded (goal-blind)  20.839     0.000  0.000     1.000
(d) hand-goal-centre+learned-spec vs (e) scripted-wrapper (R7a)      19.154     0.000  0.000     1.000
```

## Notes

- Checkpoints are selected on validation episodes of the training goals only.
- Snakes not ordered to eat starve after about 60 steps; their deaths cost nothing, so the score rewards leaving their food alone, not surviving.
- `step_limit` and `food_target` are valid GoalSpec fields but fixed at 300 and 0.
- Reproduce: `python scripts/r7b_goal_conditioned.py --from-json runs:r7b-train-* --out docs/results/r7b_goal_conditioned_16runs.md` (after the `r7b-train-*` jobs).
