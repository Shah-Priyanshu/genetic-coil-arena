# Squad: do specialists listen to the command centre? (R5)

- Date: 2026-10-23
- Commit: 8dfb52c
- Seeds: 20 run seeds (0..19) × 50 episodes each; run seed s plays episode seeds 3,000,000,000 + s × 10,000 + i. The seeds are the independent units for CIs and tests.
- CPU: Intel(R) Xeon(R) Processor @ 2.80GHz (4 logical cores visible, 2 worker processes); wall time 77 s
- Python 3.13.16
- Squad defaults: 21×21 walls, 3 snakes, 2 food per colour, health 60, T = 300, message = 1 discrete symbol, V = 8 (3 bits), sent every step, read one step later
- Communication metrics: recorded during the scored runs (no replays). Messages every 3 steps for living receivers; CIC and intervention influence on listener snapshots every 25 steps, from the specialists' exact action probabilities (`policy.probs`); MI null from 100 label shuffles.
- Live receivers (alive snakes in unfinished episodes) are the unit throughout: interventions permute, zero and resample only among them, and listener metrics average over them.
- Sanity controls: all passed

Reproduce with `python scripts/listening.py --seeds 20 --episodes 50 --workers 2 --out docs/results/listening.md`.

## Systems

- `greedy+hand-coded`: greedy specialists (chase own-colour food in their window, follow the message's direction bits when none is visible) with the hand-coded centre (`code = 2 * direction + far` of the receiver's nearest own-colour food).
- `no-comm+hand-coded`: the same centre talking to `NoCommPolicy` specialists, which never read the message. The deaf control.
- Baselines (mandatory, registered with `evaluation.register_baseline`): random actions, greedy + silent centre, greedy + random centre, no-comm greedy, and the full-view oracle.

## Interpretation

- **r = 3:** the message-following squad eats 18.6 food per episode. Every intervention costs food (positive listening: yes). Zeroing the channel costs 12.0, and the worst is shuffle_batch at 14.6. The centre uses 2.63 of 3 bits, of which MI = 2.63 bits is about the target. A message changes a specialist's action distribution by CIC = 0.86 bits per living receiver.
- **r = 5:** the message-following squad eats 26.6 food per episode. Every intervention costs food (positive listening: yes). Zeroing the channel costs 12.1, and the worst is shuffle_batch at 18.9. The centre uses 2.60 of 3 bits, of which MI = 2.60 bits is about the target. A message changes a specialist's action distribution by CIC = 0.64 bits per living receiver.
- **r = 7:** the message-following squad eats 32.6 food per episode. Every intervention costs food (positive listening: yes). Zeroing the channel costs 7.9, and the worst is shuffle_batch at 18.6. The centre uses 2.58 of 3 bits, of which MI = 2.58 bits is about the target. A message changes a specialist's action distribution by CIC = 0.42 bits per living receiver.

Reading across radii:

- The specialists listen, and only through the message. Every intervention drops team food at every radius. The deaf control's drop is exactly 0 and its CIC is 0 up to floating-point rounding (< 1e-16).
- Wrong information is mostly worse than none. shuffle_batch, swap_receivers and random_matched keep the message statistics but break the link to the receiver's situation. shuffle_batch and random_matched cost more than zeroing the channel at every radius; the exception is swap_receivers at r = 3, where the drop is no larger. Each of them leaves the squad below the message-blind `no-comm` baseline.
- Listening matters less as the window grows. CIC and the share of actions an intervention flips both fall from r = 3 to r = 7, because the greedy specialist only reads the message when no own-colour food is in view.
- The centre says about 2.6 of its 3 bits. All of that is about the target: MI(M;T) = H(M) = H(T). This holds by construction, since the code is that target. Here MI is a check that the centre is implemented correctly against an independent recomputation, not a discovery. A learned centre is where MI becomes informative. The unused ~0.4 bits are the skewed code distribution, and the 'no food' case is folded into code 0.
- p-values bottom out at 7.6e-06: with 20 paired seeds that all drop, that is the smallest two-sided Wilcoxon signed-rank p (exact, 2 / 2^20) after Holm across the four interventions. Rows up to 0.00018 are the same complete separation where tied differences make scipy use the normal approximation. The paired CIs carry the effect sizes.

## Team food per episode

Mean ± 95% t-CI over run seeds (each seed's score is the mean over its episodes).

| condition | r = 3 | r = 5 | r = 7 |
|---|---|---|---|
| greedy+hand-coded | 18.58 ± 0.77 | 26.61 ± 0.49 | 32.56 ± 0.66 |
| no-comm+hand-coded | 8.37 ± 0.28 | 15.87 ± 0.59 | 23.47 ± 0.58 |
| random-actions | 0.30 ± 0.04 | 0.30 ± 0.04 | 0.30 ± 0.04 |
| greedy+silent-centre | 6.60 ± 0.39 | 14.51 ± 0.61 | 24.69 ± 0.68 |
| greedy+random-centre | 3.88 ± 0.18 | 7.79 ± 0.38 | 13.46 ± 0.47 |
| no-comm | 8.37 ± 0.28 | 15.87 ± 0.59 | 23.47 ± 0.58 |
| oracle | 43.42 ± 0.66 | 43.42 ± 0.66 | 43.42 ± 0.66 |

## Message ablations (positive listening)

Drop = intact minus ablated team food, paired by seed; `listening` = the drop's 95% CI lies above zero. p is a two-sided Wilcoxon signed-rank test on the per-seed pairs, Holm-corrected across the four interventions per system.

| r | system | intervention | intact | ablated | drop [95% CI] | p (Holm) | A12 | listening |
|---|---|---|---|---|---|---|---|---|
| 3 | greedy+hand-coded | zero | 18.58 | 6.60 | 11.99 [11.06, 12.92] | 7.6e-06 | 1.00 | yes |
| 3 | greedy+hand-coded | shuffle_batch | 18.58 | 3.97 | 14.61 [13.84, 15.38] | 7.6e-06 | 1.00 | yes |
| 3 | greedy+hand-coded | swap_receivers | 18.58 | 7.08 | 11.50 [10.81, 12.19] | 7.6e-06 | 1.00 | yes |
| 3 | greedy+hand-coded | random_matched | 18.58 | 4.46 | 14.12 [13.46, 14.79] | 7.6e-06 | 1.00 | yes |
| 3 | no-comm+hand-coded | zero | 8.37 | 8.37 | 0.00 [0.00, 0.00] | 1.000 | 0.50 | no |
| 3 | no-comm+hand-coded | shuffle_batch | 8.37 | 8.37 | 0.00 [0.00, 0.00] | 1.000 | 0.50 | no |
| 3 | no-comm+hand-coded | swap_receivers | 8.37 | 8.37 | 0.00 [0.00, 0.00] | 1.000 | 0.50 | no |
| 3 | no-comm+hand-coded | random_matched | 8.37 | 8.37 | 0.00 [0.00, 0.00] | 1.000 | 0.50 | no |
| 5 | greedy+hand-coded | zero | 26.61 | 14.51 | 12.11 [11.37, 12.84] | 0.00018 | 1.00 | yes |
| 5 | greedy+hand-coded | shuffle_batch | 26.61 | 7.72 | 18.90 [18.38, 19.41] | 0.00018 | 1.00 | yes |
| 5 | greedy+hand-coded | swap_receivers | 26.61 | 11.92 | 14.69 [14.12, 15.26] | 7.6e-06 | 1.00 | yes |
| 5 | greedy+hand-coded | random_matched | 26.61 | 8.96 | 17.65 [17.08, 18.23] | 7.6e-06 | 1.00 | yes |
| 5 | no-comm+hand-coded | zero | 15.87 | 15.87 | 0.00 [0.00, 0.00] | 1.000 | 0.50 | no |
| 5 | no-comm+hand-coded | shuffle_batch | 15.87 | 15.87 | 0.00 [0.00, 0.00] | 1.000 | 0.50 | no |
| 5 | no-comm+hand-coded | swap_receivers | 15.87 | 15.87 | 0.00 [0.00, 0.00] | 1.000 | 0.50 | no |
| 5 | no-comm+hand-coded | random_matched | 15.87 | 15.87 | 0.00 [0.00, 0.00] | 1.000 | 0.50 | no |
| 7 | greedy+hand-coded | zero | 32.56 | 24.69 | 7.87 [7.21, 8.53] | 7.6e-06 | 1.00 | yes |
| 7 | greedy+hand-coded | shuffle_batch | 32.56 | 13.96 | 18.60 [17.75, 19.45] | 7.6e-06 | 1.00 | yes |
| 7 | greedy+hand-coded | swap_receivers | 32.56 | 18.59 | 13.97 [13.21, 14.74] | 0.00018 | 1.00 | yes |
| 7 | greedy+hand-coded | random_matched | 32.56 | 15.61 | 16.95 [16.21, 17.69] | 0.00018 | 1.00 | yes |
| 7 | no-comm+hand-coded | zero | 23.47 | 23.47 | 0.00 [0.00, 0.00] | 1.000 | 0.50 | no |
| 7 | no-comm+hand-coded | shuffle_batch | 23.47 | 23.47 | 0.00 [0.00, 0.00] | 1.000 | 0.50 | no |
| 7 | no-comm+hand-coded | swap_receivers | 23.47 | 23.47 | 0.00 [0.00, 0.00] | 1.000 | 0.50 | no |
| 7 | no-comm+hand-coded | random_matched | 23.47 | 23.47 | 0.00 [0.00, 0.00] | 1.000 | 0.50 | no |

## What the centre says (positive signalling)

Pooled over all seeds and recorded steps, living receivers only. T = the receiver's nearest own-colour food at send time, recomputed independently from the centre's observation: direction relative to heading (4) × near/far (Manhattan > 5), plus a 'no food' class. MI is Miller-Madow corrected; null q95 and p come from shuffled targets.

| r | squad | H(M) bits | used / 3-bit capacity | H(T) bits | MI(M;T) bits [null q95, p] | MI(M;direction) | MI(M;far) |
|---|---|---|---|---|---|---|---|
| 3 | greedy+hand-coded | 2.632 | 87.7% | 2.632 | 2.632 [0.000, 0.0099] | 1.910 | 0.766 |
| 3 | no-comm+hand-coded | 2.513 | 83.8% | 2.513 | 2.513 [0.000, 0.0099] | 1.959 | 0.596 |
| 3 | greedy+random-centre | 3.000 | 100.0% | 2.419 | -0.000 [0.000, 0.911] | -0.000 | 0.000 |
| 3 | greedy+silent-centre | 0.000 | 0.0% | 2.483 | 0.000 [0.000, 1.000] | 0.000 | 0.000 |
| 5 | greedy+hand-coded | 2.604 | 86.8% | 2.604 | 2.604 [0.000, 0.0099] | 1.826 | 0.829 |
| 5 | no-comm+hand-coded | 2.548 | 84.9% | 2.548 | 2.548 [0.000, 0.0099] | 1.910 | 0.703 |
| 5 | greedy+random-centre | 3.000 | 100.0% | 2.495 | -0.000 [0.000, 0.842] | -0.000 | -0.000 |
| 5 | greedy+silent-centre | 0.000 | 0.0% | 2.516 | 0.000 [0.000, 1.000] | 0.000 | 0.000 |
| 7 | greedy+hand-coded | 2.581 | 86.0% | 2.581 | 2.581 [0.000, 0.0099] | 1.746 | 0.899 |
| 7 | no-comm+hand-coded | 2.576 | 85.9% | 2.576 | 2.576 [0.000, 0.0099] | 1.847 | 0.815 |
| 7 | greedy+random-centre | 3.000 | 100.0% | 2.578 | 0.000 [0.000, 0.208] | -0.000 | -0.000 |
| 7 | greedy+silent-centre | 0.000 | 0.0% | 2.560 | 0.000 [0.000, 1.000] | 0.000 | 0.000 |

## How much messages move actions (listener side)

Per live receiver, mean ± 95% CI over seeds. CIC = I(action; message) with the situation held fixed and messages drawn from the centre's own message distribution (max log2 3 = 1.585 bits). JS = Jensen-Shannon divergence between action distributions with the real and the intervened message.

| r | squad | CIC bits | JS(shuffle_batch) bits | action changed (shuffle_batch) | JS(random_matched) bits | action changed (random_matched) |
|---|---|---|---|---|---|---|
| 3 | greedy+hand-coded | 0.856 ± 0.009 | 0.397 ± 0.007 | 0.407 ± 0.006 | 0.398 ± 0.005 | 0.410 ± 0.007 |
| 3 | no-comm+hand-coded | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| 3 | greedy+random-centre | 0.926 ± 0.011 | 0.429 ± 0.010 | 0.428 ± 0.014 | 0.428 ± 0.007 | 0.427 ± 0.010 |
| 5 | greedy+hand-coded | 0.640 ± 0.009 | 0.296 ± 0.008 | 0.303 ± 0.009 | 0.298 ± 0.006 | 0.306 ± 0.007 |
| 5 | no-comm+hand-coded | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| 5 | greedy+random-centre | 0.758 ± 0.010 | 0.358 ± 0.008 | 0.366 ± 0.010 | 0.356 ± 0.008 | 0.362 ± 0.012 |
| 7 | greedy+hand-coded | 0.424 ± 0.005 | 0.202 ± 0.004 | 0.206 ± 0.004 | 0.204 ± 0.005 | 0.208 ± 0.005 |
| 7 | no-comm+hand-coded | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| 7 | greedy+random-centre | 0.562 ± 0.011 | 0.276 ± 0.008 | 0.286 ± 0.010 | 0.272 ± 0.008 | 0.281 ± 0.011 |

## Sanity controls

| r | check | passed | detail |
|---|---|---|---|
| 3 | deaf specialists: no ablation changes team food | yes | max |drop| = 0.000 food |
| 3 | deaf specialists: CIC ~ 0 | yes | CIC = 0.0000 bits |
| 3 | random centre: MI(message; target) ~ 0 | yes | MI = -0.0001 bits (null q95 0.0001) |
| 3 | silent centre: MI(message; target) = 0 | yes | MI = 0.0000 bits |
| 3 | zero ablation == silent-centre baseline, seed by seed | yes | max |diff| = 0 |
| 5 | deaf specialists: no ablation changes team food | yes | max |drop| = 0.000 food |
| 5 | deaf specialists: CIC ~ 0 | yes | CIC = 0.0000 bits |
| 5 | random centre: MI(message; target) ~ 0 | yes | MI = -0.0001 bits (null q95 0.0001) |
| 5 | silent centre: MI(message; target) = 0 | yes | MI = 0.0000 bits |
| 5 | zero ablation == silent-centre baseline, seed by seed | yes | max |diff| = 0 |
| 7 | deaf specialists: no ablation changes team food | yes | max |drop| = 0.000 food |
| 7 | deaf specialists: CIC ~ 0 | yes | CIC = 0.0000 bits |
| 7 | random centre: MI(message; target) ~ 0 | yes | MI = 0.0001 bits (null q95 0.0001) |
| 7 | silent centre: MI(message; target) = 0 | yes | MI = 0.0000 bits |
| 7 | zero ablation == silent-centre baseline, seed by seed | yes | max |diff| = 0 |

## Harness output

`evaluation.format_table` for each radius (communication metrics are in the tables above and in `listening.json`).

### r = 3

```
Task: squad-listening-r3  (seeds: 20)
Baselines: random-actions, greedy+silent-centre, greedy+random-centre, no-comm, oracle

condition             role      n   mean ± 95% CI   IQM [95% CI]           
--------------------  --------  --  --------------  -----------------------
greedy+hand-coded     system    20  18.583 ± 0.767  18.672 [18.016, 19.340]
no-comm+hand-coded    system    20  8.374 ± 0.280   8.328 [8.148, 8.620]   
random-actions        baseline  20  0.299 ± 0.042   0.298 [0.244, 0.346]   
greedy+silent-centre  baseline  20  6.597 ± 0.394   6.536 [6.122, 7.058]   
greedy+random-centre  baseline  20  3.882 ± 0.180   3.958 [3.740, 4.108]   
no-comm               baseline  20  8.374 ± 0.280   8.328 [8.148, 8.620]   
oracle                baseline  20  43.424 ± 0.656  43.278 [42.876, 44.036]

system              intervention    ablated mean  drop [95% CI]            p (Holm)  A12    listening
------------------  --------------  ------------  -----------------------  --------  -----  ---------
greedy+hand-coded   zero            6.597         11.986 [11.056, 12.916]  0.000     1.000  yes      
greedy+hand-coded   shuffle_batch   3.969         14.614 [13.844, 15.384]  0.000     1.000  yes      
greedy+hand-coded   swap_receivers  7.083         11.500 [10.808, 12.192]  0.000     1.000  yes      
greedy+hand-coded   random_matched  4.459         14.124 [13.460, 14.788]  0.000     1.000  yes      
no-comm+hand-coded  zero            8.374         0.000 [0.000, 0.000]     1.000     0.500  no       
no-comm+hand-coded  shuffle_batch   8.374         0.000 [0.000, 0.000]     1.000     0.500  no       
no-comm+hand-coded  swap_receivers  8.374         0.000 [0.000, 0.000]     1.000     0.500  no       
no-comm+hand-coded  random_matched  8.374         0.000 [0.000, 0.000]     1.000     0.500  no       

comparison                                  mean diff  p      p (Holm)  A12  
------------------------------------------  ---------  -----  --------  -----
greedy+hand-coded vs random-actions         18.284     0.000  0.000     1.000
greedy+hand-coded vs greedy+silent-centre   11.986     0.000  0.000     1.000
greedy+hand-coded vs greedy+random-centre   14.701     0.000  0.000     1.000
greedy+hand-coded vs no-comm                10.209     0.000  0.000     1.000
greedy+hand-coded vs oracle                 -24.841    0.000  0.000     0.000
no-comm+hand-coded vs random-actions        8.075      0.000  0.000     1.000
no-comm+hand-coded vs greedy+silent-centre  1.777      0.000  0.000     0.975
no-comm+hand-coded vs greedy+random-centre  4.492      0.000  0.000     1.000
no-comm+hand-coded vs no-comm               0.000      1.000  1.000     0.500
no-comm+hand-coded vs oracle                -35.050    0.000  0.000     0.000
```

### r = 5

```
Task: squad-listening-r5  (seeds: 20)
Baselines: random-actions, greedy+silent-centre, greedy+random-centre, no-comm, oracle

condition             role      n   mean ± 95% CI   IQM [95% CI]           
--------------------  --------  --  --------------  -----------------------
greedy+hand-coded     system    20  26.613 ± 0.489  26.674 [26.090, 27.236]
no-comm+hand-coded    system    20  15.872 ± 0.589  15.868 [15.288, 16.316]
random-actions        baseline  20  0.299 ± 0.042   0.298 [0.244, 0.346]   
greedy+silent-centre  baseline  20  14.508 ± 0.609  14.442 [13.902, 15.194]
greedy+random-centre  baseline  20  7.786 ± 0.379   7.724 [7.400, 8.130]   
no-comm               baseline  20  15.872 ± 0.589  15.868 [15.288, 16.316]
oracle                baseline  20  43.424 ± 0.656  43.278 [42.876, 44.036]

system              intervention    ablated mean  drop [95% CI]            p (Holm)  A12    listening
------------------  --------------  ------------  -----------------------  --------  -----  ---------
greedy+hand-coded   zero            14.508        12.105 [11.373, 12.837]  0.000     1.000  yes      
greedy+hand-coded   shuffle_batch   7.718         18.895 [18.381, 19.409]  0.000     1.000  yes      
greedy+hand-coded   swap_receivers  11.923        14.690 [14.123, 15.257]  0.000     1.000  yes      
greedy+hand-coded   random_matched  8.962         17.651 [17.076, 18.226]  0.000     1.000  yes      
no-comm+hand-coded  zero            15.872        0.000 [0.000, 0.000]     1.000     0.500  no       
no-comm+hand-coded  shuffle_batch   15.872        0.000 [0.000, 0.000]     1.000     0.500  no       
no-comm+hand-coded  swap_receivers  15.872        0.000 [0.000, 0.000]     1.000     0.500  no       
no-comm+hand-coded  random_matched  15.872        0.000 [0.000, 0.000]     1.000     0.500  no       

comparison                                  mean diff  p      p (Holm)  A12  
------------------------------------------  ---------  -----  --------  -----
greedy+hand-coded vs random-actions         26.314     0.000  0.000     1.000
greedy+hand-coded vs greedy+silent-centre   12.105     0.000  0.000     1.000
greedy+hand-coded vs greedy+random-centre   18.827     0.000  0.000     1.000
greedy+hand-coded vs no-comm                10.741     0.000  0.000     1.000
greedy+hand-coded vs oracle                 -16.811    0.000  0.000     0.000
no-comm+hand-coded vs random-actions        15.573     0.000  0.000     1.000
no-comm+hand-coded vs greedy+silent-centre  1.364      0.004  0.007     0.770
no-comm+hand-coded vs greedy+random-centre  8.086      0.000  0.000     1.000
no-comm+hand-coded vs no-comm               0.000      1.000  1.000     0.500
no-comm+hand-coded vs oracle                -27.552    0.000  0.000     0.000
```

### r = 7

```
Task: squad-listening-r7  (seeds: 20)
Baselines: random-actions, greedy+silent-centre, greedy+random-centre, no-comm, oracle

condition             role      n   mean ± 95% CI   IQM [95% CI]           
--------------------  --------  --  --------------  -----------------------
greedy+hand-coded     system    20  32.562 ± 0.659  32.508 [31.972, 33.192]
no-comm+hand-coded    system    20  23.470 ± 0.576  23.486 [22.866, 24.054]
random-actions        baseline  20  0.299 ± 0.042   0.298 [0.244, 0.346]   
greedy+silent-centre  baseline  20  24.690 ± 0.683  24.740 [24.138, 25.410]
greedy+random-centre  baseline  20  13.465 ± 0.474  13.430 [12.982, 13.908]
no-comm               baseline  20  23.470 ± 0.576  23.486 [22.866, 24.054]
oracle                baseline  20  43.424 ± 0.656  43.278 [42.876, 44.036]

system              intervention    ablated mean  drop [95% CI]            p (Holm)  A12    listening
------------------  --------------  ------------  -----------------------  --------  -----  ---------
greedy+hand-coded   zero            24.690        7.872 [7.213, 8.531]     0.000     1.000  yes      
greedy+hand-coded   shuffle_batch   13.964        18.598 [17.746, 19.450]  0.000     1.000  yes      
greedy+hand-coded   swap_receivers  18.590        13.972 [13.207, 14.737]  0.000     1.000  yes      
greedy+hand-coded   random_matched  15.615        16.947 [16.207, 17.687]  0.000     1.000  yes      
no-comm+hand-coded  zero            23.470        0.000 [0.000, 0.000]     1.000     0.500  no       
no-comm+hand-coded  shuffle_batch   23.470        0.000 [0.000, 0.000]     1.000     0.500  no       
no-comm+hand-coded  swap_receivers  23.470        0.000 [0.000, 0.000]     1.000     0.500  no       
no-comm+hand-coded  random_matched  23.470        0.000 [0.000, 0.000]     1.000     0.500  no       

comparison                                  mean diff  p      p (Holm)  A12  
------------------------------------------  ---------  -----  --------  -----
greedy+hand-coded vs random-actions         32.263     0.000  0.000     1.000
greedy+hand-coded vs greedy+silent-centre   7.872      0.000  0.000     1.000
greedy+hand-coded vs greedy+random-centre   19.097     0.000  0.000     1.000
greedy+hand-coded vs no-comm                9.092      0.000  0.000     1.000
greedy+hand-coded vs oracle                 -10.862    0.000  0.000     0.000
no-comm+hand-coded vs random-actions        23.171     0.000  0.000     1.000
no-comm+hand-coded vs greedy+silent-centre  -1.220     0.006  0.013     0.246
no-comm+hand-coded vs greedy+random-centre  10.005     0.000  0.000     1.000
no-comm+hand-coded vs no-comm               0.000      1.000  1.000     0.500
no-comm+hand-coded vs oracle                -19.954    0.000  0.000     0.000
```
