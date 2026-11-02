# R7b: a learned command centre that follows a GoalSpec (goal G7)

**G7 verdict: KILL.** neither (b) nor (c) is better than (a) on held-out goals (CI includes 0 or below).

Scale: smoke. 2 training runs per condition; every number is a mean over training runs with a 95% CI (Student t unless stated). Mission score = eater food - 1.0 x non-eater food - death penalty (0 / 2 / 5 for low / normal / high caution) x dead eaters (see `experiments/r7b_goal_conditioned.py`). 15 training goals, 6 held-out goals: eater subset {green, blue} at every caution (`novel_subset`) and three seen-part combinations (`novel_combo`: red/high, red+blue/normal, red+green+blue/low).

Training settings: `{"radius": 3, "generations": 6, "population": 8, "reps": 1, "sigma": 0.05, "lr": 0.03, "weight_decay": 0.005, "spec_hidden": 24, "centre_hidden": 24, "eval_every": 3, "val_reps": 2, "max_steps": 300}`.
Compute: training 0.0 CPU-hours (sum over runs); evaluation 0.0 min.

## Mission score

| Condition | Seen goals | Held-out goals | Novel subset | Novel combo |
|---|---|---|---|---|
| (a) blind-centre+learned-spec | -4.10 ± 2.96 | -3.96 ± 1.59 | -4.75 ± 1.06 | -3.17 ± 2.12 |
| (b) goal-centre+learned-spec | -4.03 ± 1.27 | -3.75 ± 0.00 | -4.50 ± 0.00 | -3.00 ± 0.00 |
| (c) goal-centre+goal-spec | -4.00 ± 0.00 | -3.96 ± 0.53 | -4.67 ± 0.00 | -3.25 ± 1.06 |
| (d) hand-goal-centre+learned-spec | -4.30 ± 2.12 | -3.75 ± 1.06 | -4.50 ± 2.12 | -3.00 ± 0.00 |
| random-actions | -4.13 ± 0.85 | -3.75 ± 1.06 | -4.42 ± 1.06 | -3.08 ± 1.06 |
| greedy+hand-coded (goal-blind) | -0.20 ± 22.87 | 5.00 ± 13.77 | 6.33 ± 38.12 | 3.67 ± 10.59 |
| (e) scripted-wrapper (R7a) | -1.30 ± 4.66 | 0.58 ± 2.12 | -0.08 ± 5.29 | 1.25 ± 1.06 |

## G7

Pass: (b) or (c) beats (a) on held-out goals (Welch CI of the difference above 0, Holm-corrected Mann-Whitney p < 0.05) and reaches at least 90% of (d). Kill: neither (b) nor (c) beats (a).

(a) = -3.96, (d) = -3.75 on held-out goals.

| Condition | Held-out mean | Diff vs (a) | Welch CI | p (Holm) | Beats (a) | % of (d) |
|---|---|---|---|---|---|---|
| (b) goal-centre+learned-spec | -3.75 | +0.21 | ± 1.59 | 0.441 | no | n/a |
| (c) goal-centre+goal-spec | -3.96 | +0.00 | ± 1.11 | 1 | no | n/a |

### Pairwise differences, heldout goals

| A | B | A - B | Welch CI | p (Holm) |
|---|---|---|---|---|
| (b) goal-centre+learned-spec | (a) blind-centre+learned-spec | +0.21 | ± 1.59 | 1 |
| (c) goal-centre+goal-spec | (a) blind-centre+learned-spec | +0.00 | ± 1.11 | 1 |
| (d) hand-goal-centre+learned-spec | (a) blind-centre+learned-spec | +0.21 | ± 0.75 | 1 |
| (b) goal-centre+learned-spec | (d) hand-goal-centre+learned-spec | +0.00 | ± 1.06 | 1 |
| (c) goal-centre+goal-spec | (d) hand-goal-centre+learned-spec | -0.21 | ± 0.58 | 1 |
| (c) goal-centre+goal-spec | (b) goal-centre+learned-spec | -0.21 | ± 0.53 | 1 |
| (e) scripted-wrapper (R7a) | (a) blind-centre+learned-spec | +4.54 | ± 0.97 | 1 |
| (d) hand-goal-centre+learned-spec | (e) scripted-wrapper (R7a) | -4.33 | ± 1.15 | 1 |

### Pairwise differences, seen goals

| A | B | A - B | Welch CI | p (Holm) |
|---|---|---|---|---|
| (b) goal-centre+learned-spec | (a) blind-centre+learned-spec | +0.07 | ± 1.78 | 1 |
| (c) goal-centre+goal-spec | (a) blind-centre+learned-spec | +0.10 | ± 2.96 | 1 |
| (d) hand-goal-centre+learned-spec | (a) blind-centre+learned-spec | -0.20 | ± 1.37 | 1 |
| (b) goal-centre+learned-spec | (d) hand-goal-centre+learned-spec | +0.27 | ± 1.04 | 1 |
| (c) goal-centre+goal-spec | (d) hand-goal-centre+learned-spec | +0.30 | ± 2.12 | 1 |
| (c) goal-centre+goal-spec | (b) goal-centre+learned-spec | +0.03 | ± 1.27 | 1 |
| (e) scripted-wrapper (R7a) | (a) blind-centre+learned-spec | +2.80 | ± 2.23 | 1 |
| (d) hand-goal-centre+learned-spec | (e) scripted-wrapper (R7a) | -3.00 | ± 2.69 | 1 |

## Do the messages carry the goal?

### heldout goals (live receivers; MI in bits, Miller-Madow; null = 95th percentile of shuffled labels; probe = held-out-episode accuracy vs majority class)

| Condition | H(msg) bits | MI own-eater (null) | probe own-eater (chance) | MI caution (null) | probe caution (chance) | MI goal (null) |
|---|---|---|---|---|---|---|
| (a) blind-centre+learned-spec | 0.90 ± 7.44 | -0.003 ± 0.024 (0.006 ± 0.049) | 0.624 ± 2.932 (0.722 ± 1.684) | 0.024 ± 0.234 (0.010 ± 0.017) | 0.303 ± 0.900 (0.309 ± 0.756) | 0.047 ± 0.092 (0.028 ± 0.070) |
| (b) goal-centre+learned-spec | 1.62 ± 10.32 | 0.073 ± 0.047 (0.014 ± 0.013) | 0.618 ± 0.832 (0.664 ± 0.011) | 0.505 ± 4.816 (0.035 ± 0.077) | 0.448 ± 0.303 (0.322 ± 1.356) | 0.876 ± 8.816 (0.049 ± 0.251) |
| (c) goal-centre+goal-spec | 2.03 ± 5.82 | 0.115 ± 0.948 (0.007 ± 0.027) | 0.560 ± 1.788 (0.714 ± 0.076) | 0.466 ± 2.314 (0.014 ± 0.017) | 0.466 ± 0.699 (0.366 ± 0.196) | 0.897 ± 4.201 (0.029 ± 0.160) |
| (d) hand-goal-centre+learned-spec | 2.45 ± 1.54 | 0.868 ± 1.275 (0.014 ± 0.105) | 1.000 ± 0.000 (0.704 ± 1.301) | 0.055 ± 0.083 (0.018 ± 0.040) | 0.461 ± 0.080 (0.327 ± 0.016) | 0.147 ± 0.071 (0.048 ± 0.101) |

### seen goals (live receivers; MI in bits, Miller-Madow; null = 95th percentile of shuffled labels; probe = held-out-episode accuracy vs majority class)

| Condition | H(msg) bits | MI own-eater (null) | probe own-eater (chance) | MI caution (null) | probe caution (chance) | MI goal (null) |
|---|---|---|---|---|---|---|
| (a) blind-centre+learned-spec | 0.82 ± 6.83 | 0.001 ± 0.037 (0.013 ± 0.006) | 0.519 ± 1.058 (0.523 ± 1.100) | 0.017 ± 0.118 (0.016 ± 0.021) | 0.165 ± 0.250 (0.172 ± 0.166) | 0.090 ± 0.446 (0.058 ± 0.108) |
| (b) goal-centre+learned-spec | 1.60 ± 5.25 | 0.070 ± 0.030 (0.008 ± 0.028) | 0.528 ± 0.109 (0.558 ± 0.710) | 0.454 ± 2.997 (0.015 ± 0.066) | 0.515 ± 2.705 (0.131 ± 0.613) | 0.849 ± 5.688 (0.100 ± 0.120) |
| (c) goal-centre+goal-spec | 2.00 ± 2.33 | 0.125 ± 1.271 (0.011 ± 0.041) | 0.562 ± 0.926 (0.509 ± 0.197) | 0.335 ± 0.591 (0.013 ± 0.002) | 0.315 ± 0.697 (0.182 ± 0.184) | 0.932 ± 2.637 (0.061 ± 0.070) |
| (d) hand-goal-centre+learned-spec | 2.26 ± 0.60 | 1.001 ± 0.012 (0.006 ± 0.031) | 1.000 ± 0.000 (0.367 ± 0.409) | 0.050 ± 0.352 (0.009 ± 0.038) | 0.239 ± 1.351 (0.193 ± 0.391) | 0.466 ± 0.038 (0.084 ± 0.007) |

## Ablations (score drop, intact minus intervened, paired over training runs)

Message interventions are the harness's; `goals swapped` tells the policies the goal of another episode (a different eater set) while the score uses the true goal.

### heldout goals

| Condition | random_matched | shuffle_batch | swap_receivers | zero | goals swapped |
|---|---|---|---|---|---|
| (a) blind-centre+learned-spec | -0.04 ± 0.53 | -0.08 ± 1.06 | -0.08 ± 1.06 | -0.08 ± 1.06 | +0.00 ± 0.00 |
| (b) goal-centre+learned-spec | +0.08 ± 1.06 | +0.08 ± 1.06 | +0.08 ± 0.00 | +0.08 ± 0.00 | +0.04 ± 0.53 |
| (c) goal-centre+goal-spec | +0.00 ± 0.00 | -0.25 ± 1.06 | -0.17 ± 1.06 | -0.04 ± 2.65 | -0.33 ± 4.24 |
| (d) hand-goal-centre+learned-spec | +0.12 ± 0.53 | +0.08 ± 1.06 | +0.04 ± 0.53 | +0.04 ± 0.53 | +0.08 ± 0.00 |

### seen goals

| Condition | random_matched | shuffle_batch | swap_receivers | zero | goals swapped |
|---|---|---|---|---|---|
| (a) blind-centre+learned-spec | +0.00 ± 0.00 | +0.00 ± 0.00 | +0.03 ± 0.42 | -0.07 ± 1.69 | +0.00 ± 0.00 |
| (b) goal-centre+learned-spec | +0.03 ± 0.42 | +0.03 ± 0.42 | -0.00 ± 0.85 | +0.13 ± 0.00 | +0.07 ± 0.85 |
| (c) goal-centre+goal-spec | +0.07 ± 0.85 | +0.03 ± 0.42 | +0.03 ± 2.12 | +0.03 ± 1.27 | -0.00 ± 0.85 |
| (d) hand-goal-centre+learned-spec | -0.13 ± 0.85 | -0.10 ± 0.42 | -0.10 ± 0.42 | -0.10 ± 0.42 | +0.00 ± 0.85 |

## Per-goal mean mission score (mean over training runs)

### seen

| Goal | (a) blind-centre+learned-spec | (b) goal-centre+learned-spec | (c) goal-centre+goal-spec | (d) hand-goal-centre+learned-spec | random-actions | greedy+hand-coded (goal-blind) | (e) scripted-wrapper (R7a) |
|---|---|---|---|---|---|---|---|
| r/low | 0.0 | 0.5 | 0.5 | 0.0 | 0.0 | -7.5 | 2.0 |
| r/normal | -2.0 | -2.0 | -2.5 | -2.5 | -2.5 | -14.0 | -0.5 |
| g/low | -0.5 | 0.0 | 0.0 | -0.5 | 0.0 | -7.5 | 7.0 |
| g/normal | -3.0 | -2.0 | -2.0 | -2.5 | -2.0 | -10.5 | -2.0 |
| g/high | -4.5 | -4.5 | -4.5 | -4.5 | -5.0 | -2.5 | 1.5 |
| r+g/low | 0.0 | 0.0 | 0.0 | 0.5 | 0.0 | 7.5 | 1.0 |
| r+g/normal | -3.0 | -4.0 | -4.0 | -4.0 | -4.0 | 1.0 | -2.0 |
| r+g/high | -10.0 | -9.5 | -10.0 | -9.5 | -9.5 | 7.5 | -7.0 |
| b/low | -1.0 | -0.5 | 0.0 | -1.5 | -0.5 | -7.5 | -0.5 |
| b/normal | -2.0 | -2.0 | -2.0 | -2.0 | -2.0 | -9.0 | -1.5 |
| b/high | -5.0 | -5.5 | -5.0 | -6.5 | -6.0 | -11.0 | -4.0 |
| r+b/low | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 10.0 | 7.5 |
| r+b/high | -10.5 | -10.5 | -10.0 | -10.5 | -10.5 | 13.0 | -8.5 |
| r+g+b/normal | -5.5 | -5.5 | -5.5 | -6.0 | -6.0 | 14.5 | -0.5 |
| r+g+b/high | -14.5 | -15.0 | -15.0 | -15.0 | -14.0 | 13.0 | -12.0 |

### heldout

| Goal | (a) blind-centre+learned-spec | (b) goal-centre+learned-spec | (c) goal-centre+goal-spec | (d) hand-goal-centre+learned-spec | random-actions | greedy+hand-coded (goal-blind) | (e) scripted-wrapper (R7a) |
|---|---|---|---|---|---|---|---|
| g+b/low | 0.0 | 0.0 | -0.2 | 0.0 | 0.0 | 9.2 | 4.8 |
| g+b/normal | -4.0 | -3.8 | -3.5 | -4.0 | -3.5 | 7.5 | 3.0 |
| g+b/high | -10.2 | -9.8 | -10.2 | -9.5 | -9.8 | 2.2 | -8.0 |
| r/high | -5.2 | -5.0 | -5.0 | -5.0 | -5.0 | 1.0 | -4.5 |
| r+b/normal | -4.2 | -4.0 | -4.8 | -4.2 | -4.2 | -8.2 | 2.8 |
| r+g+b/low | 0.0 | 0.0 | 0.0 | 0.2 | 0.0 | 18.2 | 5.5 |

## Harness table, heldout goals

```
Task: squad-goal-heldout  (seeds: 2)
Baselines: random-actions, greedy+hand-coded (goal-blind), (e) scripted-wrapper (R7a)

condition                          role      n  mean ± 95% CI   IQM [95% CI]           
---------------------------------  --------  -  --------------  -----------------------
(a) blind-centre+learned-spec      system    2  -3.958 ± 1.588  -3.958 [-4.083, -3.833]
(b) goal-centre+learned-spec       system    2  -3.750 ± 0.000  -3.750 [-3.750, -3.750]
(c) goal-centre+goal-spec          system    2  -3.958 ± 0.529  -3.958 [-4.000, -3.917]
(d) hand-goal-centre+learned-spec  system    2  -3.750 ± 1.059  -3.750 [-3.833, -3.667]
random-actions                     baseline  2  -3.750 ± 1.059  -3.750 [-3.833, -3.667]
greedy+hand-coded (goal-blind)     baseline  2  5.000 ± 13.765  5.000 [3.917, 6.083]   
(e) scripted-wrapper (R7a)         baseline  2  0.583 ± 2.118   0.583 [0.417, 0.750]   

system                             intervention    ablated mean  drop [95% CI]           p (Holm)  A12    listening
---------------------------------  --------------  ------------  ----------------------  --------  -----  ---------
(a) blind-centre+learned-spec      zero            -3.875        -0.083 [-1.142, 0.976]  1.000     0.375  no       
(a) blind-centre+learned-spec      shuffle_batch   -3.875        -0.083 [-1.142, 0.976]  1.000     0.375  no       
(a) blind-centre+learned-spec      swap_receivers  -3.875        -0.083 [-1.142, 0.976]  1.000     0.375  no       
(a) blind-centre+learned-spec      random_matched  -3.917        -0.042 [-0.571, 0.488]  1.000     0.375  no       
(b) goal-centre+learned-spec       zero            -3.833        0.083 [0.083, 0.083]    1.000     1.000  yes      
(b) goal-centre+learned-spec       shuffle_batch   -3.833        0.083 [-0.976, 1.142]   1.000     0.750  no       
(b) goal-centre+learned-spec       swap_receivers  -3.833        0.083 [0.083, 0.083]    1.000     1.000  yes      
(b) goal-centre+learned-spec       random_matched  -3.833        0.083 [-0.976, 1.142]   1.000     0.750  no       
(c) goal-centre+goal-spec          zero            -3.917        -0.042 [-2.689, 2.605]  1.000     0.500  no       
(c) goal-centre+goal-spec          shuffle_batch   -3.708        -0.250 [-1.309, 0.809]  1.000     0.000  no       
(c) goal-centre+goal-spec          swap_receivers  -3.792        -0.167 [-1.226, 0.892]  1.000     0.000  no       
(c) goal-centre+goal-spec          random_matched  -3.958        0.000 [0.000, 0.000]    1.000     0.500  no       
(d) hand-goal-centre+learned-spec  zero            -3.792        0.042 [-0.488, 0.571]   1.000     0.625  no       
(d) hand-goal-centre+learned-spec  shuffle_batch   -3.833        0.083 [-0.976, 1.142]   1.000     0.750  no       
(d) hand-goal-centre+learned-spec  swap_receivers  -3.792        0.042 [-0.488, 0.571]   1.000     0.625  no       
(d) hand-goal-centre+learned-spec  random_matched  -3.875        0.125 [-0.404, 0.654]   1.000     0.875  no       

comparison                                                           mean diff  p      p (Holm)  A12  
-------------------------------------------------------------------  ---------  -----  --------  -----
(a) blind-centre+learned-spec vs random-actions                      -0.208     0.414  1.000     0.125
(a) blind-centre+learned-spec vs greedy+hand-coded (goal-blind)      -8.958     0.333  1.000     0.000
(a) blind-centre+learned-spec vs (e) scripted-wrapper (R7a)          -4.542     0.333  1.000     0.000
(b) goal-centre+learned-spec vs random-actions                       0.000      1.000  1.000     0.500
(b) goal-centre+learned-spec vs greedy+hand-coded (goal-blind)       -8.750     0.221  1.000     0.000
(b) goal-centre+learned-spec vs (e) scripted-wrapper (R7a)           -4.333     0.221  1.000     0.000
(c) goal-centre+goal-spec vs random-actions                          -0.208     0.333  1.000     0.000
(c) goal-centre+goal-spec vs greedy+hand-coded (goal-blind)          -8.958     0.333  1.000     0.000
(c) goal-centre+goal-spec vs (e) scripted-wrapper (R7a)              -4.542     0.333  1.000     0.000
(d) hand-goal-centre+learned-spec vs random-actions                  0.000      1.000  1.000     0.500
(d) hand-goal-centre+learned-spec vs greedy+hand-coded (goal-blind)  -8.750     0.333  1.000     0.000
(d) hand-goal-centre+learned-spec vs (e) scripted-wrapper (R7a)      -4.333     0.333  1.000     0.000
```

## Harness table, seen goals

```
Task: squad-goal-seen  (seeds: 2)
Baselines: random-actions, greedy+hand-coded (goal-blind), (e) scripted-wrapper (R7a)

condition                          role      n  mean ± 95% CI    IQM [95% CI]           
---------------------------------  --------  -  ---------------  -----------------------
(a) blind-centre+learned-spec      system    2  -4.100 ± 2.965   -4.100 [-4.333, -3.867]
(b) goal-centre+learned-spec       system    2  -4.033 ± 1.271   -4.033 [-4.133, -3.933]
(c) goal-centre+goal-spec          system    2  -4.000 ± 0.000   -4.000 [-4.000, -4.000]
(d) hand-goal-centre+learned-spec  system    2  -4.300 ± 2.118   -4.300 [-4.467, -4.133]
random-actions                     baseline  2  -4.133 ± 0.847   -4.133 [-4.200, -4.067]
greedy+hand-coded (goal-blind)     baseline  2  -0.200 ± 22.871  -0.200 [-2.000, 1.600] 
(e) scripted-wrapper (R7a)         baseline  2  -1.300 ± 4.659   -1.300 [-1.667, -0.933]

system                             intervention    ablated mean  drop [95% CI]           p (Holm)  A12    listening
---------------------------------  --------------  ------------  ----------------------  --------  -----  ---------
(a) blind-centre+learned-spec      zero            -4.033        -0.067 [-1.761, 1.627]  1.000     0.500  no       
(a) blind-centre+learned-spec      shuffle_batch   -4.100        0.000 [0.000, 0.000]    1.000     0.500  no       
(a) blind-centre+learned-spec      swap_receivers  -4.133        0.033 [-0.390, 0.457]   1.000     0.625  no       
(a) blind-centre+learned-spec      random_matched  -4.100        0.000 [0.000, 0.000]    1.000     0.500  no       
(b) goal-centre+learned-spec       zero            -4.167        0.133 [0.133, 0.133]    1.000     0.750  yes      
(b) goal-centre+learned-spec       shuffle_batch   -4.067        0.033 [-0.390, 0.457]   1.000     0.625  no       
(b) goal-centre+learned-spec       swap_receivers  -4.033        -0.000 [-0.847, 0.847]  1.000     0.500  no       
(b) goal-centre+learned-spec       random_matched  -4.067        0.033 [-0.390, 0.457]   1.000     0.625  no       
(c) goal-centre+goal-spec          zero            -4.033        0.033 [-1.237, 1.304]   1.000     0.500  no       
(c) goal-centre+goal-spec          shuffle_batch   -4.033        0.033 [-0.390, 0.457]   1.000     0.750  no       
(c) goal-centre+goal-spec          swap_receivers  -4.033        0.033 [-2.084, 2.151]   1.000     0.500  no       
(c) goal-centre+goal-spec          random_matched  -4.067        0.067 [-0.780, 0.914]   1.000     0.750  no       
(d) hand-goal-centre+learned-spec  zero            -4.200        -0.100 [-0.524, 0.324]  1.000     0.250  no       
(d) hand-goal-centre+learned-spec  shuffle_batch   -4.200        -0.100 [-0.524, 0.324]  1.000     0.250  no       
(d) hand-goal-centre+learned-spec  swap_receivers  -4.200        -0.100 [-0.524, 0.324]  1.000     0.250  no       
(d) hand-goal-centre+learned-spec  random_matched  -4.167        -0.133 [-0.980, 0.714]  1.000     0.250  no       

comparison                                                           mean diff  p      p (Holm)  A12  
-------------------------------------------------------------------  ---------  -----  --------  -----
(a) blind-centre+learned-spec vs random-actions                      0.033      1.000  1.000     0.500
(a) blind-centre+learned-spec vs greedy+hand-coded (goal-blind)      -3.900     0.333  1.000     0.000
(a) blind-centre+learned-spec vs (e) scripted-wrapper (R7a)          -2.800     0.333  1.000     0.000
(b) goal-centre+learned-spec vs random-actions                       0.100      0.667  1.000     0.750
(b) goal-centre+learned-spec vs greedy+hand-coded (goal-blind)       -3.833     0.333  1.000     0.000
(b) goal-centre+learned-spec vs (e) scripted-wrapper (R7a)           -2.733     0.333  1.000     0.000
(c) goal-centre+goal-spec vs random-actions                          0.133      0.221  1.000     1.000
(c) goal-centre+goal-spec vs greedy+hand-coded (goal-blind)          -3.800     0.221  1.000     0.000
(c) goal-centre+goal-spec vs (e) scripted-wrapper (R7a)              -2.700     0.221  1.000     0.000
(d) hand-goal-centre+learned-spec vs random-actions                  -0.167     0.667  1.000     0.250
(d) hand-goal-centre+learned-spec vs greedy+hand-coded (goal-blind)  -4.100     0.333  1.000     0.000
(d) hand-goal-centre+learned-spec vs (e) scripted-wrapper (R7a)      -3.000     0.333  1.000     0.000
```

## Notes

- Checkpoints are selected on validation episodes of the training goals only.
- Snakes not ordered to eat starve after about 60 steps; their deaths cost nothing, so the score rewards leaving their food alone, not surviving.
- `step_limit` and `food_target` are valid GoalSpec fields but fixed at 300 and 0.
- Reproduce: `python scripts/r7b_goal_conditioned.py --from-json runs:r7b-train-* --out docs/results/report.md` (after the `r7b-train-*` jobs).
