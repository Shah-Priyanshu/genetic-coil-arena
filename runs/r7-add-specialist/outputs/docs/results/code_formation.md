# Squad: helping a learned code form (R6 follow-up)

- Date: 2026-10-23
- Commit: 39929cc (code; this report is committed on top of it)
- Compute: Intel(R) Xeon(R) Processor @ 2.10GHz, 4 logical cores, 4 worker processes, numpy/numba only. Training wall time 33.2 min (1.99 CPU-hours), evaluation 0.3 min.
- Training: condition (c) `learned-centre+learned-spec` of `learned_protocol` (joint ES of centre and specialists, r = 3), 6 runs (run indices 0–5) per arm, 600 generations × population 64 × 8 episodes per member; all other settings as in `learned_protocol.md`. Run s of every arm starts from the same weights and sees the same ES noise and training episodes, so arms are compared paired by run.
- Held-out evaluation: run s plays test episodes 3,500,000,000 + s × 10,000 + i, i < 100 (the same seeds as `learned_protocol.md`).
- Arms: `baseline` (R6 method); `entropy` (fitness += w·H(M), w from 1 to 0 linearly over the first 50% of generations); `temperature` (centre samples from softmax(logits / T) in training, T from 1 to 0 over the first 50%; argmax for validation and test).
- **Code formed** (per run): H(M) > 0.5 bits on the held-out episodes and the `shuffle_batch` drop (intact − shuffled team food, paired t-CI over the 100 episodes) has a 95% lower bound > 0. Shuffling, not zeroing, is the test: for learned listeners a zeroed message is a wrong message, so it costs food even when the code carries no information (R6's failed run lost 5.4 food to zeroing but 0.1 to shuffling).

Reproduce with `python scripts/code_formation.py --runs 6 --generations 600 --population 64 --episodes-per-member 8 --entropy-bonus 1 --entropy-anneal 0.5 --centre-temperature 1 --temperature-anneal 0.5 --eval-episodes 100 --workers 4 --out docs/results/code_formation.md`.

## Interpretation

**Verdict: both aids made a code form in every run (6/6 each, against 4/6 for the baseline), at no extra compute. The entropy bonus also learned fastest and ended highest. With n = 6 per arm, none of these differences passes a test. Read this as a strong pilot signal, not as an established effect.**

- **Code formation.** The baseline reproduced R6's failure. In run 2 (the same run index that failed in R6) the centre went silent (H(M) = 0.00, shuffle drop 0.0). In run 3 it sent 0.48 bits that nobody used (shuffle drop 0.3, CI includes 0). Both runs ate 6 to 7 food, the silent level. With either aid, both runs formed a working code: entropy got 26.8 and 22.2 food, temperature 21.1 and 24.4. The 95% Clopper-Pearson interval for the aids is [0.54, 1.00] against [0.22, 0.96] for the baseline. Fisher's test (unpaired, conservative) gives p = 0.45 per arm, 0.91 after Holm. Six runs cannot separate 6/6 from 4/6.
- **Caveat on "never".** In R6's 1000-generation run, run 3 formed its code late, at about generation 750, so at 600 generations the baseline's run 3 may be late rather than failed. A late code is also costly: R6's run 3 ended at 20.2 food against 28 to 31 for the early-forming runs. Either way, the aids move code formation earlier.
- **Team food of (c).**
  - Entropy: 25.6 ± 2.1. Temperature: 23.0 ± 2.0. Baseline: 18.8 ± 10.2.
  - The paired differences from the baseline are +6.7 [−2.8, +16.2] for entropy and +4.1 [−6.3, +14.6] for temperature. Wilcoxon after Holm gives p = 0.19 and 0.69.
  - In runs where a code formed, the entropy arm matches the baseline (25.6 vs 25.1), so the bonus does not cost final quality. The temperature arm's codes are slightly worse (23.0).
  - For scale, (c) reached 22.5 ± 14.8 after 1000 generations in R6. The hand-coded listener condition (b) reached 26.1 on validation at generation 500 and 30.7 on test at 1000.
- **Speed.**
  - Entropy: the mean validation curve is ahead of the baseline from generation 200 on (17.2 vs 12.2 food at generation 300, and 23.0 vs 15.7 at 400). It plateaus near 25 by generation 500.
  - Temperature: it starts slower (3.2 vs 6.8 at generation 200), because sampled codes are noisy messages while T is high. It catches up once T reaches 0 at generation 300.
- **What the codes say.**
  - Entropy codes carry more: H(M) = 1.45 ± 0.25 bits and MI(M;T) = 0.79 ± 0.07, against 1.09 / 0.62 for temperature and 1.0 to 1.5 / about 0.7 for the baseline runs that formed a code.
  - Under the bonus, H(M) rises to 2.7 bits by generation 100, then falls back to about 1.3 to 1.5 once the bonus anneals away. The bonus keeps the centre's output input-dependent while the listeners learn to read it, and the team reward then prunes the code to what is useful.
  - Every arm still encodes direction, not distance. That is the same coarse code R6 found.
- **Mechanism (plausible, not tested).** Weight decay shrinks a centre whose messages nobody reads, and its argmax then collapses onto one bias-determined symbol. The entropy bonus pays directly for input-dependence. For an argmax centre, H(M) equals I(M; centre input), so the bonus is exactly a positive-signalling term. The temperature aid instead smooths the ES objective over the centre's logits.
- **Compute and caveats.**
  - Only 6 runs per arm, one setting per aid (bonus weight 1 annealed over 300 generations; T = 1 annealed over 300), r = 3, and 600 generations.
  - The planned 10, then 8, runs per arm were cut to 6 after two container restarts. The per-run cache kept every finished run, but the runs in progress at each restart were lost, about 15 minutes of extra wall time beyond the 33 minutes reported above.
  - No hyperparameter of either aid was tuned.

**Recommendation for the next GPU run.** Enable the entropy bonus: `--entropy-bonus 1`, with the annealing horizon kept near the roughly 300 generations tested here. For a 2000-generation run that is `--entropy-anneal 0.15`. `--entropy-anneal 0.5` would also be reasonable but is untested at 1000 generations. The bonus is cheaper and better on every measure here than the temperature aid. Run it as a separate job, `--modes learned --entropy-bonus 1 --entropy-anneal 0.15`, next to the queued unmodified `r6-c-r3` job, so that the GPU run measures both the formation rate and whether (c) closes the gap to (b) at n = 10.

## Code formation and team food (held-out)

Mean ± 95% t-CI over training runs. References on the same test seeds: scripted `greedy+hand-coded` 18.89 ± 0.55, no-comm 8.15 ± 0.34, oracle 43.07 ± 0.64.

| arm | code formed | 95% CI (Clopper-Pearson) | team food (c) | team food, runs with a code | H(M) bits | MI(M;T) bits | shuffle drop |
|---|---|---|---|---|---|---|---|
| baseline | 4/6 | [0.22, 0.96] | 18.83 ± 10.21 | 25.05 ± 2.42 | 0.87 ± 0.57 | 0.45 ± 0.36 | 15.33 ± 12.41 |
| entropy | 6/6 | [0.54, 1.00] | 25.55 ± 2.06 | 25.55 ± 2.06 | 1.45 ± 0.25 | 0.79 ± 0.07 | 23.17 ± 1.69 |
| temperature | 6/6 | [0.54, 1.00] | 22.96 ± 1.97 | 22.96 ± 1.97 | 1.09 ± 0.21 | 0.62 ± 0.11 | 20.80 ± 1.88 |

Paired by run; Holm across the arms compared with the baseline. The Fisher test ignores the pairing (conservative).

| arm vs baseline | team food diff [95% paired t-CI] | p (Wilcoxon, Holm) | p (Fisher, code formed, Holm) |
|---|---|---|---|
| entropy | +6.72 [-2.78, +16.22] | 0.188 | 0.909 |
| temperature | +4.13 [-6.34, +14.59] | 0.688 | 0.909 |

## Run by run

team food / H(M) bits / shuffle drop [95% lower bound] / code formed.

| run | baseline: food / H(M) / shuffle drop [low] / formed | entropy: food / H(M) / shuffle drop [low] / formed | temperature: food / H(M) / shuffle drop [low] / formed |
|---|---|---|---|
| 0 | 24.8 / 1.12 / 22.7 [21.5] / yes | 25.3 / 1.49 / 22.8 [21.2] / yes | 22.8 / 0.99 / 20.8 [19.4] / yes |
| 1 | 23.2 / 1.54 / 20.9 [19.5] / yes | 25.9 / 1.49 / 23.9 [22.5] / yes | 25.9 / 1.02 / 23.4 [21.9] / yes |
| 2 | 7.1 / 0.00 / 0.0 [0.0] / no | 26.8 / 1.67 / 23.8 [22.3] / yes | 21.1 / 0.99 / 19.8 [18.4] / yes |
| 3 | 5.7 / 0.48 / 0.3 [-0.5] / no | 22.2 / 0.99 / 20.2 [18.7] / yes | 24.4 / 1.00 / 22.5 [21.3] / yes |
| 4 | 25.4 / 1.08 / 23.8 [22.5] / yes | 28.1 / 1.53 / 24.9 [23.3] / yes | 22.0 / 1.49 / 19.0 [17.5] / yes |
| 5 | 26.9 / 1.00 / 24.2 [22.9] / yes | 25.0 / 1.55 / 23.5 [22.1] / yes | 21.5 / 1.07 / 19.4 [18.2] / yes |

## Learning curves (validation, ES mean, argmax centre; mean over runs)

Team food:

| arm | gen 0 | gen 100 | gen 200 | gen 300 | gen 400 | gen 500 | gen 600 |
|---|---|---|---|---|---|---|---|
| baseline | 0.31 | 1.52 | 6.83 | 12.18 | 15.74 | 17.66 | 19.40 |
| entropy | 0.31 | 1.57 | 8.62 | 17.22 | 23.04 | 25.06 | 25.89 |
| temperature | 0.31 | 1.11 | 3.19 | 9.29 | 16.40 | 21.59 | 24.05 |

H(M) bits of the centre's symbols on the validation episodes:

| arm | gen 0 | gen 100 | gen 200 | gen 300 | gen 400 | gen 500 | gen 600 |
|---|---|---|---|---|---|---|---|
| baseline | 1.42 | 0.85 | 0.81 | 0.75 | 0.76 | 0.89 | 0.87 |
| entropy | 1.42 | 2.66 | 1.76 | 1.44 | 1.42 | 1.49 | 1.27 |
| temperature | 1.42 | 1.15 | 0.98 | 1.17 | 1.17 | 1.08 | 1.09 |

## Harness output

`evaluation.format_table` per arm (condition (c) with the scripted reference and the registered baselines, message ablations, communication metrics).

### baseline

```
Task: squad-learned-protocol-r3  (seeds: 6)
Baselines: random-actions, greedy+silent-centre, greedy+random-centre, no-comm, oracle

condition                    role      n  mean ± 95% CI    IQM [95% CI]           
---------------------------  --------  -  ---------------  -----------------------
learned-centre+learned-spec  system    6  18.830 ± 10.206  20.108 [10.405, 25.742]
greedy+hand-coded            system    6  18.890 ± 0.554   19.002 [18.430, 19.240]
random-actions               baseline  6  0.325 ± 0.072    0.302 [0.285, 0.388]   
greedy+silent-centre         baseline  6  6.680 ± 0.453    6.612 [6.355, 7.088]   
greedy+random-centre         baseline  6  3.593 ± 0.203    3.567 [3.452, 3.763]   
no-comm                      baseline  6  8.153 ± 0.336    8.140 [7.890, 8.435]   
oracle                       baseline  6  43.072 ± 0.637   42.945 [42.667, 43.620]

system                       intervention    ablated mean  drop [95% CI]           p (Holm)  A12    listening
---------------------------  --------------  ------------  ----------------------  --------  -----  ---------
learned-centre+learned-spec  zero            2.607         16.223 [4.860, 27.586]  0.125     0.972  yes      
learned-centre+learned-spec  shuffle_batch   3.523         15.307 [2.921, 27.692]  0.188     0.958  yes      
learned-centre+learned-spec  swap_receivers  6.535         12.295 [2.311, 22.279]  0.188     0.819  yes      
learned-centre+learned-spec  random_matched  3.607         15.223 [3.079, 27.367]  0.188     0.958  yes      

comparison                                           mean diff  p      p (Holm)  A12  
---------------------------------------------------  ---------  -----  --------  -----
learned-centre+learned-spec vs random-actions        18.505     0.005  0.022     1.000
learned-centre+learned-spec vs greedy+silent-centre  12.150     0.093  0.186     0.806
learned-centre+learned-spec vs greedy+random-centre  15.237     0.002  0.022     1.000
learned-centre+learned-spec vs no-comm               10.677     0.394  0.394     0.667
learned-centre+learned-spec vs oracle                -24.242    0.002  0.022     0.000
greedy+hand-coded vs random-actions                  18.565     0.005  0.022     1.000
greedy+hand-coded vs greedy+silent-centre            12.210     0.002  0.022     1.000
greedy+hand-coded vs greedy+random-centre            15.297     0.002  0.022     1.000
greedy+hand-coded vs no-comm                         10.737     0.002  0.022     1.000
greedy+hand-coded vs oracle                          -24.182    0.002  0.022     0.000

system                       changed_random_matched  changed_random_matched_ci  changed_shuffle_batch  changed_shuffle_batch_ci  cic_bits  cic_bits_ci  distinct_codes  distinct_codes_ci  js_random_matched_bits  js_random_matched_bits_ci  js_shuffle_batch_bits  js_shuffle_batch_bits_ci  message_entropy_bits  message_entropy_bits_ci  mi_bits  mi_bits_ci  mi_direction_bits  mi_direction_bits_ci  mi_far_bits  mi_far_bits_ci  mi_null_q95_bits  mi_null_q95_bits_ci  mi_p_value  mi_p_value_ci  target_entropy_bits  target_entropy_bits_ci  topsim  topsim_ci  utilization  utilization_ci
---------------------------  ----------------------  -------------------------  ---------------------  ------------------------  --------  -----------  --------------  -----------------  ----------------------  -------------------------  ---------------------  ------------------------  --------------------  -----------------------  -------  ----------  -----------------  --------------------  -----------  --------------  ----------------  -------------------  ----------  -------------  -------------------  ----------------------  ------  ---------  -----------  --------------
learned-centre+learned-spec  0.211                   0.116                      0.213                  0.115                     0.281     0.222        3.500           1.848              0.143                   0.095                      0.145                  0.095                     0.869                 0.569                    0.453    0.359       0.430              0.347                 0.010        0.016           0.001             0.000                0.175       0.424          2.697                0.113                   0.216   0.203      0.290        0.190         
greedy+hand-coded            0.415                   0.012                      0.407                  0.013                     0.857     0.010        8.000           0.000              0.401                   0.013                      0.393                  0.010                     2.635                 0.006                    2.636    0.006       1.911              0.007                 0.769        0.008           0.001             0.000                0.010       0.000          2.635                0.006                   0.747   0.007      0.878        0.002         
```

### entropy

```
Task: squad-learned-protocol-r3  (seeds: 6)
Baselines: random-actions, greedy+silent-centre, greedy+random-centre, no-comm, oracle

condition                    role      n  mean ± 95% CI   IQM [95% CI]           
---------------------------  --------  -  --------------  -----------------------
learned-centre+learned-spec  system    6  25.550 ± 2.062  25.755 [23.688, 27.039]
greedy+hand-coded            system    6  18.890 ± 0.554  19.002 [18.430, 19.240]
random-actions               baseline  6  0.325 ± 0.072   0.302 [0.285, 0.388]   
greedy+silent-centre         baseline  6  6.680 ± 0.453   6.612 [6.355, 7.088]   
greedy+random-centre         baseline  6  3.593 ± 0.203   3.567 [3.452, 3.763]   
no-comm                      baseline  6  8.153 ± 0.336   8.140 [7.890, 8.435]   
oracle                       baseline  6  43.072 ± 0.637  42.945 [42.667, 43.620]

system                       intervention    ablated mean  drop [95% CI]            p (Holm)  A12    listening
---------------------------  --------------  ------------  -----------------------  --------  -----  ---------
learned-centre+learned-spec  zero            2.498         23.052 [21.469, 24.634]  0.125     1.000  yes      
learned-centre+learned-spec  shuffle_batch   2.387         23.163 [21.426, 24.900]  0.125     1.000  yes      
learned-centre+learned-spec  swap_receivers  7.582         17.968 [15.913, 20.024]  0.125     1.000  yes      
learned-centre+learned-spec  random_matched  2.505         23.045 [21.477, 24.613]  0.125     1.000  yes      

comparison                                           mean diff  p      p (Holm)  A12  
---------------------------------------------------  ---------  -----  --------  -----
learned-centre+learned-spec vs random-actions        25.225     0.005  0.022     1.000
learned-centre+learned-spec vs greedy+silent-centre  18.870     0.002  0.022     1.000
learned-centre+learned-spec vs greedy+random-centre  21.957     0.002  0.022     1.000
learned-centre+learned-spec vs no-comm               17.397     0.002  0.022     1.000
learned-centre+learned-spec vs oracle                -17.522    0.002  0.022     0.000
greedy+hand-coded vs random-actions                  18.565     0.005  0.022     1.000
greedy+hand-coded vs greedy+silent-centre            12.210     0.002  0.022     1.000
greedy+hand-coded vs greedy+random-centre            15.297     0.002  0.022     1.000
greedy+hand-coded vs no-comm                         10.737     0.002  0.022     1.000
greedy+hand-coded vs oracle                          -24.182    0.002  0.022     0.000

system                       changed_random_matched  changed_random_matched_ci  changed_shuffle_batch  changed_shuffle_batch_ci  cic_bits  cic_bits_ci  distinct_codes  distinct_codes_ci  js_random_matched_bits  js_random_matched_bits_ci  js_shuffle_batch_bits  js_shuffle_batch_bits_ci  message_entropy_bits  message_entropy_bits_ci  mi_bits  mi_bits_ci  mi_direction_bits  mi_direction_bits_ci  mi_far_bits  mi_far_bits_ci  mi_null_q95_bits  mi_null_q95_bits_ci  mi_p_value  mi_p_value_ci  target_entropy_bits  target_entropy_bits_ci  topsim  topsim_ci  utilization  utilization_ci
---------------------------  ----------------------  -------------------------  ---------------------  ------------------------  --------  -----------  --------------  -----------------  ----------------------  -------------------------  ---------------------  ------------------------  --------------------  -----------------------  -------  ----------  -----------------  --------------------  -----------  --------------  ----------------  -------------------  ----------  -------------  -------------------  ----------------------  ------  ---------  -----------  --------------
learned-centre+learned-spec  0.286                   0.028                      0.283                  0.029                     0.414     0.070        4.333           1.434              0.198                   0.033                      0.196                  0.031                     1.453                 0.247                    0.792    0.070       0.741              0.072                 0.020        0.022           0.001             0.000                0.010       0.000          2.723                0.039                   0.276   0.075      0.484        0.082         
greedy+hand-coded            0.415                   0.012                      0.407                  0.013                     0.857     0.010        8.000           0.000              0.401                   0.013                      0.393                  0.010                     2.635                 0.006                    2.636    0.006       1.911              0.007                 0.769        0.008           0.001             0.000                0.010       0.000          2.635                0.006                   0.747   0.007      0.878        0.002         
```

### temperature

```
Task: squad-learned-protocol-r3  (seeds: 6)
Baselines: random-actions, greedy+silent-centre, greedy+random-centre, no-comm, oracle

condition                    role      n  mean ± 95% CI   IQM [95% CI]           
---------------------------  --------  -  --------------  -----------------------
learned-centre+learned-spec  system    6  22.957 ± 1.966  22.670 [21.425, 24.775]
greedy+hand-coded            system    6  18.890 ± 0.554  19.002 [18.430, 19.240]
random-actions               baseline  6  0.325 ± 0.072   0.302 [0.285, 0.388]   
greedy+silent-centre         baseline  6  6.680 ± 0.453   6.612 [6.355, 7.088]   
greedy+random-centre         baseline  6  3.593 ± 0.203   3.567 [3.452, 3.763]   
no-comm                      baseline  6  8.153 ± 0.336   8.140 [7.890, 8.435]   
oracle                       baseline  6  43.072 ± 0.637  42.945 [42.667, 43.620]

system                       intervention    ablated mean  drop [95% CI]            p (Holm)  A12    listening
---------------------------  --------------  ------------  -----------------------  --------  -----  ---------
learned-centre+learned-spec  zero            2.882         20.075 [18.179, 21.971]  0.125     1.000  yes      
learned-centre+learned-spec  shuffle_batch   2.162         20.795 [18.977, 22.613]  0.125     1.000  yes      
learned-centre+learned-spec  swap_receivers  6.417         16.540 [14.768, 18.312]  0.125     1.000  yes      
learned-centre+learned-spec  random_matched  2.245         20.712 [18.948, 22.476]  0.125     1.000  yes      

comparison                                           mean diff  p      p (Holm)  A12  
---------------------------------------------------  ---------  -----  --------  -----
learned-centre+learned-spec vs random-actions        22.632     0.005  0.022     1.000
learned-centre+learned-spec vs greedy+silent-centre  16.277     0.002  0.022     1.000
learned-centre+learned-spec vs greedy+random-centre  19.363     0.002  0.022     1.000
learned-centre+learned-spec vs no-comm               14.803     0.002  0.022     1.000
learned-centre+learned-spec vs oracle                -20.115    0.002  0.022     0.000
greedy+hand-coded vs random-actions                  18.565     0.005  0.022     1.000
greedy+hand-coded vs greedy+silent-centre            12.210     0.002  0.022     1.000
greedy+hand-coded vs greedy+random-centre            15.297     0.002  0.022     1.000
greedy+hand-coded vs no-comm                         10.737     0.002  0.022     1.000
greedy+hand-coded vs oracle                          -24.182    0.002  0.022     0.000

system                       changed_random_matched  changed_random_matched_ci  changed_shuffle_batch  changed_shuffle_batch_ci  cic_bits  cic_bits_ci  distinct_codes  distinct_codes_ci  js_random_matched_bits  js_random_matched_bits_ci  js_shuffle_batch_bits  js_shuffle_batch_bits_ci  message_entropy_bits  message_entropy_bits_ci  mi_bits  mi_bits_ci  mi_direction_bits  mi_direction_bits_ci  mi_far_bits  mi_far_bits_ci  mi_null_q95_bits  mi_null_q95_bits_ci  mi_p_value  mi_p_value_ci  target_entropy_bits  target_entropy_bits_ci  topsim  topsim_ci  utilization  utilization_ci
---------------------------  ----------------------  -------------------------  ---------------------  ------------------------  --------  -----------  --------------  -----------------  ----------------------  -------------------------  ---------------------  ------------------------  --------------------  -----------------------  -------  ----------  -----------------  --------------------  -----------  --------------  ----------------  -------------------  ----------  -------------  -------------------  ----------------------  ------  ---------  -----------  --------------
learned-centre+learned-spec  0.263                   0.039                      0.258                  0.041                     0.405     0.036        3.667           1.084              0.185                   0.021                      0.185                  0.022                     1.092                 0.206                    0.622    0.112       0.581              0.101                 0.014        0.025           0.001             0.000                0.010       0.000          2.727                0.051                   0.297   0.047      0.364        0.069         
greedy+hand-coded            0.415                   0.012                      0.407                  0.013                     0.857     0.010        8.000           0.000              0.401                   0.013                      0.393                  0.010                     2.635                 0.006                    2.636    0.006       1.911              0.007                 0.769        0.008           0.001             0.000                0.010       0.000          2.635                0.006                   0.747   0.007      0.878        0.002         
```
