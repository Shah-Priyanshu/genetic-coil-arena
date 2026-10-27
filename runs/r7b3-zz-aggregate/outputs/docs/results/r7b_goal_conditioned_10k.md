# R7b: a learned command centre that follows a GoalSpec (goal G7)

**G7 verdict: PASS.** a goal-conditioned centre beats (a) and reaches 90% of (d) on held-out goals.

Scale: full. 8 training runs per condition; every number is a mean over training runs with a 95% CI (Student t unless stated). Mission score = eater food - 1.0 x non-eater food - death penalty (0 / 2 / 5 for low / normal / high caution) x dead eaters (see `experiments/r7b_goal_conditioned.py`). 15 training goals, 6 held-out goals: eater subset {green, blue} at every caution (`novel_subset`) and three seen-part combinations (`novel_combo`: red/high, red+blue/normal, red+green+blue/low).

Training settings: `{"radius": 3, "generations": 10000, "population": 64, "reps": 1, "sigma": 0.05, "lr": 0.03, "weight_decay": 0.005, "spec_hidden": 24, "centre_hidden": 24, "eval_every": 50, "val_reps": 2, "max_steps": 300}`.
Compute: training 374.0 CPU-hours (sum over runs); evaluation 1.3 min.

## Mission score

| Condition | Seen goals | Held-out goals | Novel subset | Novel combo |
|---|---|---|---|---|
| (a) blind-centre+learned-spec | -3.97 ± 0.08 | -3.37 ± 0.16 | -4.23 ± 0.17 | -2.51 ± 0.16 |
| (b) goal-centre+learned-spec | 18.43 ± 1.24 | 21.37 ± 1.42 | 21.04 ± 1.77 | 21.70 ± 1.62 |
| (c) goal-centre+goal-spec | 18.86 ± 1.33 | 21.57 ± 1.93 | 21.57 ± 1.93 | 21.58 ± 2.20 |
| (d) hand-goal-centre+learned-spec | 19.72 ± 1.59 | 23.49 ± 2.23 | 23.48 ± 2.48 | 23.50 ± 2.10 |
| random-actions | -4.05 ± 0.02 | -3.73 ± 0.05 | -4.55 ± 0.05 | -2.91 ± 0.06 |
| greedy+hand-coded (goal-blind) | -1.68 ± 0.91 | 3.56 ± 0.72 | 2.94 ± 0.86 | 4.18 ± 0.82 |
| (e) scripted-wrapper (R7a) | -0.24 ± 0.22 | 0.78 ± 0.38 | -0.04 ± 0.38 | 1.61 ± 0.64 |

## G7

Pass: (b) or (c) beats (a) on held-out goals (Welch CI of the difference above 0, Holm-corrected Mann-Whitney p < 0.05) and reaches at least 90% of (d). Kill: neither (b) nor (c) beats (a).

(a) = -3.37, (d) = 23.49 on held-out goals.

| Condition | Held-out mean | Diff vs (a) | Welch CI | p (Holm) | Beats (a) | % of (d) |
|---|---|---|---|---|---|---|
| (b) goal-centre+learned-spec | 21.37 | +24.74 | ± 1.42 | 0.000311 | yes | 91% |
| (c) goal-centre+goal-spec | 21.57 | +24.94 | ± 1.93 | 0.000311 | yes | 92% |

### Pairwise differences, heldout goals

| A | B | A - B | Welch CI | p (Holm) |
|---|---|---|---|---|
| (b) goal-centre+learned-spec | (a) blind-centre+learned-spec | +24.74 | ± 1.42 | 0.00124 |
| (c) goal-centre+goal-spec | (a) blind-centre+learned-spec | +24.94 | ± 1.93 | 0.00124 |
| (d) hand-goal-centre+learned-spec | (a) blind-centre+learned-spec | +26.86 | ± 2.23 | 0.00124 |
| (b) goal-centre+learned-spec | (d) hand-goal-centre+learned-spec | -2.12 | ± 2.44 | 0.15 |
| (c) goal-centre+goal-spec | (d) hand-goal-centre+learned-spec | -1.91 | ± 2.68 | 0.21 |
| (c) goal-centre+goal-spec | (b) goal-centre+learned-spec | +0.21 | ± 2.19 | 0.959 |
| (e) scripted-wrapper (R7a) | (a) blind-centre+learned-spec | +4.15 | ± 0.39 | 0.00124 |
| (d) hand-goal-centre+learned-spec | (e) scripted-wrapper (R7a) | +22.70 | ± 2.24 | 0.00124 |

### Pairwise differences, seen goals

| A | B | A - B | Welch CI | p (Holm) |
|---|---|---|---|---|
| (b) goal-centre+learned-spec | (a) blind-centre+learned-spec | +22.40 | ± 1.24 | 0.00738 |
| (c) goal-centre+goal-spec | (a) blind-centre+learned-spec | +22.83 | ± 1.33 | 0.00738 |
| (d) hand-goal-centre+learned-spec | (a) blind-centre+learned-spec | +23.69 | ± 1.59 | 0.00738 |
| (b) goal-centre+learned-spec | (d) hand-goal-centre+learned-spec | -1.29 | ± 1.84 | 0.391 |
| (c) goal-centre+goal-spec | (d) hand-goal-centre+learned-spec | -0.86 | ± 1.89 | 0.688 |
| (c) goal-centre+goal-spec | (b) goal-centre+learned-spec | +0.43 | ± 1.65 | 0.721 |
| (e) scripted-wrapper (R7a) | (a) blind-centre+learned-spec | +3.73 | ± 0.23 | 0.00738 |
| (d) hand-goal-centre+learned-spec | (e) scripted-wrapper (R7a) | +19.96 | ± 1.59 | 0.00738 |

## Do the messages carry the goal?

### heldout goals (live receivers; MI in bits, Miller-Madow; null = 95th percentile of shuffled labels; probe = held-out-episode accuracy vs majority class)

| Condition | H(msg) bits | MI own-eater (null) | probe own-eater (chance) | MI caution (null) | probe caution (chance) | MI goal (null) |
|---|---|---|---|---|---|---|
| (a) blind-centre+learned-spec | 1.32 ± 0.38 | 0.000 ± 0.000 (0.001 ± 0.000) | 0.676 ± 0.008 (0.676 ± 0.008) | 0.001 ± 0.001 (0.001 ± 0.000) | 0.334 ± 0.009 (0.329 ± 0.014) | 0.002 ± 0.001 (0.001 ± 0.000) |
| (b) goal-centre+learned-spec | 1.09 ± 0.12 | 0.130 ± 0.074 (0.000 ± 0.000) | 0.915 ± 0.041 (0.901 ± 0.022) | 0.017 ± 0.014 (0.000 ± 0.000) | 0.350 ± 0.059 (0.282 ± 0.047) | 0.028 ± 0.019 (0.000 ± 0.000) |
| (c) goal-centre+goal-spec | 0.99 ± 0.07 | 0.040 ± 0.034 (0.000 ± 0.000) | 0.900 ± 0.020 (0.899 ± 0.020) | 0.016 ± 0.020 (0.000 ± 0.000) | 0.310 ± 0.051 (0.340 ± 0.045) | 0.026 ± 0.029 (0.000 ± 0.000) |
| (d) hand-goal-centre+learned-spec | 2.46 ± 0.04 | 0.514 ± 0.021 (0.000 ± 0.000) | 1.000 ± 0.000 (0.885 ± 0.006) | 0.025 ± 0.003 (0.000 ± 0.000) | 0.370 ± 0.055 (0.375 ± 0.050) | 0.059 ± 0.005 (0.001 ± 0.000) |

### seen goals (live receivers; MI in bits, Miller-Madow; null = 95th percentile of shuffled labels; probe = held-out-episode accuracy vs majority class)

| Condition | H(msg) bits | MI own-eater (null) | probe own-eater (chance) | MI caution (null) | probe caution (chance) | MI goal (null) |
|---|---|---|---|---|---|---|
| (a) blind-centre+learned-spec | 1.32 ± 0.38 | 0.000 ± 0.000 (0.000 ± 0.000) | 0.516 ± 0.029 (0.536 ± 0.015) | 0.001 ± 0.001 (0.001 ± 0.000) | 0.330 ± 0.010 (0.334 ± 0.012) | 0.006 ± 0.003 (0.003 ± 0.001) |
| (b) goal-centre+learned-spec | 1.14 ± 0.14 | 0.210 ± 0.115 (0.000 ± 0.000) | 0.879 ± 0.061 (0.855 ± 0.034) | 0.005 ± 0.004 (0.000 ± 0.000) | 0.360 ± 0.010 (0.358 ± 0.011) | 0.036 ± 0.025 (0.001 ± 0.000) |
| (c) goal-centre+goal-spec | 0.99 ± 0.04 | 0.051 ± 0.045 (0.000 ± 0.000) | 0.858 ± 0.030 (0.857 ± 0.029) | 0.007 ± 0.007 (0.000 ± 0.000) | 0.350 ± 0.006 (0.349 ± 0.009) | 0.016 ± 0.010 (0.001 ± 0.000) |
| (d) hand-goal-centre+learned-spec | 2.46 ± 0.03 | 0.677 ± 0.013 (0.000 ± 0.000) | 1.000 ± 0.000 (0.823 ± 0.005) | 0.004 ± 0.001 (0.000 ± 0.000) | 0.366 ± 0.013 (0.346 ± 0.018) | 0.092 ± 0.004 (0.001 ± 0.000) |

## Ablations (score drop, intact minus intervened, paired over training runs)

Message interventions are the harness's; `goals swapped` tells the policies the goal of another episode (a different eater set) while the score uses the true goal.

### heldout goals

| Condition | random_matched | shuffle_batch | swap_receivers | zero | goals swapped |
|---|---|---|---|---|---|
| (a) blind-centre+learned-spec | +0.16 ± 0.11 | +0.16 ± 0.11 | +0.11 ± 0.11 | +0.16 ± 0.17 | +0.00 ± 0.00 |
| (b) goal-centre+learned-spec | +24.47 ± 1.29 | +24.69 ± 1.45 | +21.25 ± 1.26 | +24.84 ± 1.47 | +25.61 ± 1.69 |
| (c) goal-centre+goal-spec | +24.49 ± 1.87 | +24.50 ± 1.86 | +18.92 ± 0.82 | +24.01 ± 2.80 | +25.96 ± 3.09 |
| (d) hand-goal-centre+learned-spec | +26.62 ± 2.17 | +26.82 ± 2.20 | +23.29 ± 2.21 | +27.18 ± 2.21 | +28.81 ± 2.98 |

### seen goals

| Condition | random_matched | shuffle_batch | swap_receivers | zero | goals swapped |
|---|---|---|---|---|---|
| (a) blind-centre+learned-spec | +0.05 ± 0.13 | -0.01 ± 0.07 | +0.03 ± 0.09 | +0.07 ± 0.09 | +0.00 ± 0.00 |
| (b) goal-centre+learned-spec | +22.07 ± 1.20 | +22.24 ± 1.26 | +19.23 ± 1.29 | +22.43 ± 1.26 | +23.14 ± 1.27 |
| (c) goal-centre+goal-spec | +22.17 ± 1.10 | +22.39 ± 1.23 | +16.54 ± 2.14 | +22.30 ± 2.19 | +23.22 ± 1.31 |
| (d) hand-goal-centre+learned-spec | +23.44 ± 1.65 | +23.58 ± 1.65 | +20.87 ± 1.43 | +23.76 ± 1.61 | +23.84 ± 1.78 |

## Per-goal mean mission score (mean over training runs)

### seen

| Goal | (a) blind-centre+learned-spec | (b) goal-centre+learned-spec | (c) goal-centre+goal-spec | (d) hand-goal-centre+learned-spec | random-actions | greedy+hand-coded (goal-blind) | (e) scripted-wrapper (R7a) |
|---|---|---|---|---|---|---|---|
| r/low | -0.5 | 14.5 | 15.2 | 17.1 | -0.1 | -4.3 | 2.0 |
| r/normal | -2.7 | 13.1 | 14.9 | 15.5 | -2.1 | -8.6 | 0.6 |
| g/low | -0.7 | 14.7 | 15.0 | 16.0 | -0.1 | -6.0 | 2.4 |
| g/normal | -2.5 | 13.4 | 15.4 | 15.9 | -2.1 | -7.6 | 0.5 |
| g/high | -5.2 | 13.9 | 13.7 | 15.4 | -5.2 | -9.7 | -2.9 |
| r+g/low | 0.4 | 25.0 | 23.6 | 25.7 | 0.1 | 7.9 | 3.7 |
| r+g/normal | -3.2 | 22.3 | 22.4 | 23.4 | -3.9 | 2.7 | 1.4 |
| r+g/high | -9.5 | 20.0 | 20.5 | 21.1 | -9.9 | -0.8 | -6.1 |
| b/low | -0.6 | 15.7 | 16.6 | 15.1 | -0.1 | -8.5 | 2.5 |
| b/normal | -2.5 | 15.1 | 15.4 | 15.5 | -2.1 | -7.5 | 0.0 |
| b/high | -5.5 | 14.0 | 13.9 | 15.8 | -5.0 | -10.3 | -2.6 |
| r+b/low | 0.3 | 25.1 | 26.3 | 24.9 | 0.1 | 8.8 | 5.0 |
| r+b/high | -9.5 | 19.4 | 19.2 | 20.3 | -10.0 | -1.2 | -5.2 |
| r+g+b/normal | -4.6 | 27.3 | 27.5 | 30.4 | -5.7 | 14.2 | 2.7 |
| r+g+b/high | -13.4 | 23.0 | 23.3 | 23.5 | -14.8 | 5.6 | -7.7 |

### heldout

| Goal | (a) blind-centre+learned-spec | (b) goal-centre+learned-spec | (c) goal-centre+goal-spec | (d) hand-goal-centre+learned-spec | random-actions | greedy+hand-coded (goal-blind) | (e) scripted-wrapper (R7a) |
|---|---|---|---|---|---|---|---|
| g+b/low | 0.3 | 23.6 | 23.7 | 26.5 | 0.2 | 5.5 | 4.2 |
| g+b/normal | -3.5 | 21.5 | 22.2 | 23.8 | -3.9 | 3.8 | 1.2 |
| g+b/high | -9.5 | 18.0 | 18.9 | 20.2 | -9.9 | -0.5 | -5.5 |
| r/high | -5.3 | 12.2 | 12.0 | 14.6 | -5.1 | -9.7 | -2.9 |
| r+b/normal | -3.6 | 22.0 | 21.7 | 23.2 | -3.9 | 3.9 | 1.3 |
| r+g+b/low | 1.3 | 30.9 | 31.0 | 32.7 | 0.2 | 18.3 | 6.4 |

## Harness table, heldout goals

```
Task: squad-goal-heldout  (seeds: 8)
Baselines: random-actions, greedy+hand-coded (goal-blind), (e) scripted-wrapper (R7a)

condition                          role      n  mean ± 95% CI   IQM [95% CI]           
---------------------------------  --------  -  --------------  -----------------------
(a) blind-centre+learned-spec      system    8  -3.370 ± 0.156  -3.383 [-3.521, -3.217]
(b) goal-centre+learned-spec       system    8  21.369 ± 1.418  21.392 [20.073, 22.869]
(c) goal-centre+goal-spec          system    8  21.574 ± 1.926  21.146 [19.923, 23.268]
(d) hand-goal-centre+learned-spec  system    8  23.489 ± 2.228  24.087 [21.900, 25.567]
random-actions                     baseline  8  -3.731 ± 0.047  -3.737 [-3.779, -3.679]
greedy+hand-coded (goal-blind)     baseline  8  3.562 ± 0.715   3.604 [2.840, 4.279]   
(e) scripted-wrapper (R7a)         baseline  8  0.784 ± 0.383   0.715 [0.435, 1.085]   

system                             intervention    ablated mean  drop [95% CI]            p (Holm)  A12    listening
---------------------------------  --------------  ------------  -----------------------  --------  -----  ---------
(a) blind-centre+learned-spec      zero            -3.531        0.161 [-0.008, 0.331]    0.070     0.750  no       
(a) blind-centre+learned-spec      shuffle_batch   -3.531        0.161 [0.055, 0.267]     0.062     0.750  yes      
(a) blind-centre+learned-spec      swap_receivers  -3.484        0.115 [0.009, 0.220]     0.070     0.719  yes      
(a) blind-centre+learned-spec      random_matched  -3.532        0.162 [0.050, 0.275]     0.070     0.766  yes      
(b) goal-centre+learned-spec       zero            -3.474        24.843 [23.370, 26.316]  0.031     1.000  yes      
(b) goal-centre+learned-spec       shuffle_batch   -3.318        24.686 [23.236, 26.137]  0.031     1.000  yes      
(b) goal-centre+learned-spec       swap_receivers  0.118         21.251 [19.994, 22.508]  0.031     1.000  yes      
(b) goal-centre+learned-spec       random_matched  -3.102        24.471 [23.176, 25.765]  0.031     1.000  yes      
(c) goal-centre+goal-spec          zero            -2.434        24.008 [21.210, 26.807]  0.031     1.000  yes      
(c) goal-centre+goal-spec          shuffle_batch   -2.922        24.496 [22.637, 26.355]  0.031     1.000  yes      
(c) goal-centre+goal-spec          swap_receivers  2.657         18.917 [18.099, 19.735]  0.031     1.000  yes      
(c) goal-centre+goal-spec          random_matched  -2.918        24.492 [22.621, 26.362]  0.031     1.000  yes      
(d) hand-goal-centre+learned-spec  zero            -3.695        27.183 [24.973, 29.393]  0.031     1.000  yes      
(d) hand-goal-centre+learned-spec  shuffle_batch   -3.334        26.823 [24.626, 29.020]  0.031     1.000  yes      
(d) hand-goal-centre+learned-spec  swap_receivers  0.198         23.291 [21.076, 25.505]  0.031     1.000  yes      
(d) hand-goal-centre+learned-spec  random_matched  -3.129        26.618 [24.446, 28.790]  0.031     1.000  yes      

comparison                                                           mean diff  p      p (Holm)  A12  
-------------------------------------------------------------------  ---------  -----  --------  -----
(a) blind-centre+learned-spec vs random-actions                      0.361      0.001  0.004     1.000
(a) blind-centre+learned-spec vs greedy+hand-coded (goal-blind)      -6.932     0.000  0.002     0.000
(a) blind-centre+learned-spec vs (e) scripted-wrapper (R7a)          -4.154     0.000  0.002     0.000
(b) goal-centre+learned-spec vs random-actions                       25.100     0.001  0.004     1.000
(b) goal-centre+learned-spec vs greedy+hand-coded (goal-blind)       17.806     0.000  0.002     1.000
(b) goal-centre+learned-spec vs (e) scripted-wrapper (R7a)           20.584     0.000  0.002     1.000
(c) goal-centre+goal-spec vs random-actions                          25.305     0.001  0.004     1.000
(c) goal-centre+goal-spec vs greedy+hand-coded (goal-blind)          18.011     0.000  0.002     1.000
(c) goal-centre+goal-spec vs (e) scripted-wrapper (R7a)              20.790     0.000  0.002     1.000
(d) hand-goal-centre+learned-spec vs random-actions                  27.220     0.001  0.004     1.000
(d) hand-goal-centre+learned-spec vs greedy+hand-coded (goal-blind)  19.926     0.000  0.002     1.000
(d) hand-goal-centre+learned-spec vs (e) scripted-wrapper (R7a)      22.704     0.000  0.002     1.000
```

## Harness table, seen goals

```
Task: squad-goal-seen  (seeds: 8)
Baselines: random-actions, greedy+hand-coded (goal-blind), (e) scripted-wrapper (R7a)

condition                          role      n  mean ± 95% CI   IQM [95% CI]           
---------------------------------  --------  -  --------------  -----------------------
(a) blind-centre+learned-spec      system    8  -3.970 ± 0.075  -3.975 [-4.040, -3.883]
(b) goal-centre+learned-spec       system    8  18.429 ± 1.243  18.273 [17.406, 19.452]
(c) goal-centre+goal-spec          system    8  18.859 ± 1.329  18.744 [17.508, 20.071]
(d) hand-goal-centre+learned-spec  system    8  19.716 ± 1.591  20.110 [18.471, 21.242]
random-actions                     baseline  8  -4.054 ± 0.017  -4.058 [-4.071, -4.035]
greedy+hand-coded (goal-blind)     baseline  8  -1.680 ± 0.905  -1.606 [-2.760, -0.744]
(e) scripted-wrapper (R7a)         baseline  8  -0.243 ± 0.224  -0.179 [-0.394, -0.092]

system                             intervention    ablated mean  drop [95% CI]            p (Holm)  A12    listening
---------------------------------  --------------  ------------  -----------------------  --------  -----  ---------
(a) blind-centre+learned-spec      zero            -4.035        0.066 [-0.023, 0.154]    0.812     0.703  no       
(a) blind-centre+learned-spec      shuffle_batch   -3.965        -0.005 [-0.075, 0.064]   1.000     0.500  no       
(a) blind-centre+learned-spec      swap_receivers  -3.999        0.029 [-0.061, 0.119]    1.000     0.586  no       
(a) blind-centre+learned-spec      random_matched  -4.022        0.052 [-0.080, 0.184]    0.891     0.688  no       
(b) goal-centre+learned-spec       zero            -3.999        22.428 [21.170, 23.686]  0.031     1.000  yes      
(b) goal-centre+learned-spec       shuffle_batch   -3.806        22.235 [20.975, 23.495]  0.031     1.000  yes      
(b) goal-centre+learned-spec       swap_receivers  -0.802        19.231 [17.944, 20.518]  0.031     1.000  yes      
(b) goal-centre+learned-spec       random_matched  -3.645        22.074 [20.876, 23.272]  0.031     1.000  yes      
(c) goal-centre+goal-spec          zero            -3.444        22.303 [20.113, 24.493]  0.031     1.000  yes      
(c) goal-centre+goal-spec          shuffle_batch   -3.529        22.389 [21.163, 23.614]  0.031     1.000  yes      
(c) goal-centre+goal-spec          swap_receivers  2.315         16.545 [14.405, 18.685]  0.031     1.000  yes      
(c) goal-centre+goal-spec          random_matched  -3.315        22.174 [21.069, 23.279]  0.031     1.000  yes      
(d) hand-goal-centre+learned-spec  zero            -4.041        23.756 [22.145, 25.367]  0.031     1.000  yes      
(d) hand-goal-centre+learned-spec  shuffle_batch   -3.865        23.580 [21.932, 25.228]  0.031     1.000  yes      
(d) hand-goal-centre+learned-spec  swap_receivers  -1.152        20.868 [19.441, 22.294]  0.031     1.000  yes      
(d) hand-goal-centre+learned-spec  random_matched  -3.724        23.440 [21.786, 25.093]  0.031     1.000  yes      

comparison                                                           mean diff  p      p (Holm)  A12  
-------------------------------------------------------------------  ---------  -----  --------  -----
(a) blind-centre+learned-spec vs random-actions                      0.084      0.040  0.040     0.812
(a) blind-centre+learned-spec vs greedy+hand-coded (goal-blind)      -2.290     0.001  0.008     0.000
(a) blind-centre+learned-spec vs (e) scripted-wrapper (R7a)          -3.727     0.001  0.008     0.000
(b) goal-centre+learned-spec vs random-actions                       22.483     0.001  0.008     1.000
(b) goal-centre+learned-spec vs greedy+hand-coded (goal-blind)       20.109     0.000  0.002     1.000
(b) goal-centre+learned-spec vs (e) scripted-wrapper (R7a)           18.672     0.001  0.008     1.000
(c) goal-centre+goal-spec vs random-actions                          22.914     0.001  0.008     1.000
(c) goal-centre+goal-spec vs greedy+hand-coded (goal-blind)          20.540     0.000  0.002     1.000
(c) goal-centre+goal-spec vs (e) scripted-wrapper (R7a)              19.102     0.001  0.008     1.000
(d) hand-goal-centre+learned-spec vs random-actions                  23.770     0.001  0.008     1.000
(d) hand-goal-centre+learned-spec vs greedy+hand-coded (goal-blind)  21.396     0.000  0.002     1.000
(d) hand-goal-centre+learned-spec vs (e) scripted-wrapper (R7a)      19.958     0.001  0.008     1.000
```

## Notes

- Checkpoints are selected on validation episodes of the training goals only.
- Snakes not ordered to eat starve after about 60 steps; their deaths cost nothing, so the score rewards leaving their food alone, not surviving.
- `step_limit` and `food_target` are valid GoalSpec fields but fixed at 300 and 0.
- Reproduce: `python scripts/r7b_goal_conditioned.py --from-json runs:r7b-train-* --out docs/results/r7b_goal_conditioned_10k.md` (after the `r7b-train-*` jobs).
