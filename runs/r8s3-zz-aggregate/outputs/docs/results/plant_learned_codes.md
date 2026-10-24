# Plant floor: learned codes against hand-designed encodings (R8, step 3, goal G2)

- Provenance: commits 5450962e5; scale `full`; 12 partial result files; 99 CPU-minutes of tuning and cell work; 10 replicates with results.
- Protocol and data: those of step 2 (`plant_bytes_baselines_rieth.json`): Rieth et al. 2017 runs, 17 diagnosable faults, window 20 samples, one uplink per 10, 10 disjoint run-level replicates (the same runs as step 2, so differences are paired by replicate), centre calibrated on validation runs to 5% false alarms. Every number is mean ± 95% t-CI over replicates (a replicate's value averages its encoder seeds).
- Learned codes: each node encodes its local window (per variable mean, std, slope, min, max and the log energy of 3 FFT bands, standardised on the training windows) into exactly `budget` bytes; the centre is step 2's calibrated MLP on the decoded messages, stacking the last 6 uplinks, with the settings step 2 chose on the tuning block for the hand features (hidden 64, L2 1e-3, 200 iterations). Encoder hyperparameters were chosen on the tuning block only.
  - **L1 end-to-end**: per-node MLP, `8 B` sign bits through a straight-through (or logistic-noise binary-concrete) bottleneck, trained jointly with a classifier head by class-balanced cross-entropy (torch, CPU); the head is then discarded and the centre is re-trained on the frozen code.
  - **L2 residual k-means**: per node, PCA of the local features and `B` stages of 256-centroid k-means on the residual; one byte per stage; no labels.
  - **L3 local classifier**: each node trains its own fault classifier and sends its top `B` classes, one byte each: 5 bits of class id and 3 bits of posterior. The centre trains on cross-fitted messages (classifiers that never saw the run), because a classifier is over-confident on its own training runs.

## Verdict (goal G2)

G2 passes if, at 1, 2 or 4 B per uplink, a learned code beats the best hand design's macro-F1 at the calibrated 5% false-alarm rate with the 95% CI of the paired difference over 10 replicates excluding 0; it is killed if no budget shows a significant win.

**PASS.** A learned code beats the best hand design: e2e at 1 B: +0.141 ± 0.008 over features; vq at 1 B: +0.009 ± 0.007 over features; local at 1 B: +0.147 ± 0.008 over features; e2e at 2 B: +0.123 ± 0.006 over features; vq at 2 B: +0.049 ± 0.005 over features; local at 2 B: +0.128 ± 0.006 over features; e2e at 4 B: +0.061 ± 0.003 over features; vq at 4 B: +0.031 ± 0.003 over features; local at 4 B: +0.064 ± 0.003 over features.

- Largest difference: local at 1 B, +0.147 ± 0.008 (macro-F1 0.969 against (features) 0.823).
- Multiple comparisons: 9 tests are run (methods x budgets); 9 survive Holm correction at 5%. The G2 rule as written uses the unadjusted 95% CI; read a single marginal win with that in mind.
- Re-thresholded to exactly 5% test false alarms (not deployable, removes calibration error), the verdict is unchanged.

## Learned against the best hand design (macro-F1 at the calibrated threshold)

| Method | Budget | Learned | Best hand design | Difference (learned - hand) | p | p (Holm) | n | 95% CI excludes 0 above |
|---|---|---|---|---|---|---|---|---|
| (L1) end-to-end encoder + centre, binary bottleneck | 1 B | 0.964 | (features) 0.823 | +0.141 ± 0.008 | 1.79e-11 | 7.17e-11 | 10 | yes |
| (L2) residual k-means codebook (unsupervised) | 1 B | 0.832 | (features) 0.823 | +0.009 ± 0.007 | 0.0138 | 0.0138 | 10 | yes |
| (L3) local classifier, quantised top posteriors | 1 B | 0.969 | (features) 0.823 | +0.147 ± 0.008 | 1.23e-11 | 6.13e-11 | 10 | yes |
| (L1) end-to-end encoder + centre, binary bottleneck | 2 B | 0.965 | (features) 0.842 | +0.123 ± 0.006 | 6.82e-12 | 5.46e-11 | 10 | yes |
| (L2) residual k-means codebook (unsupervised) | 2 B | 0.890 | (features) 0.842 | +0.049 ± 0.005 | 6.82e-09 | 1.36e-08 | 10 | yes |
| (L3) local classifier, quantised top posteriors | 2 B | 0.970 | (features) 0.842 | +0.128 ± 0.006 | 4.23e-12 | 3.8e-11 | 10 | yes |
| (L1) end-to-end encoder + centre, binary bottleneck | 4 B | 0.967 | (features) 0.905 | +0.061 ± 0.003 | 8.85e-12 | 5.46e-11 | 10 | yes |
| (L2) residual k-means codebook (unsupervised) | 4 B | 0.936 | (features) 0.905 | +0.031 ± 0.003 | 3.49e-09 | 1.05e-08 | 10 | yes |
| (L3) local classifier, quantised top posteriors | 4 B | 0.970 | (features) 0.905 | +0.064 ± 0.003 | 7.58e-12 | 5.46e-11 | 10 | yes |

### Same comparison at exactly 5% test false alarms

| Method | Budget | Learned | Best hand design | Difference (learned - hand) | p | p (Holm) | n | 95% CI excludes 0 above |
|---|---|---|---|---|---|---|---|---|
| (L1) end-to-end encoder + centre, binary bottleneck | 1 B | 0.963 | (features) 0.823 | +0.141 ± 0.008 | 2.07e-11 | 8.33e-11 | 10 | yes |
| (L2) residual k-means codebook (unsupervised) | 1 B | 0.833 | (features) 0.823 | +0.010 ± 0.007 | 0.0105 | 0.0105 | 10 | yes |
| (L3) local classifier, quantised top posteriors | 1 B | 0.970 | (features) 0.823 | +0.147 ± 0.008 | 1.67e-11 | 8.33e-11 | 10 | yes |
| (L1) end-to-end encoder + centre, binary bottleneck | 2 B | 0.965 | (features) 0.842 | +0.123 ± 0.006 | 5.98e-12 | 3.91e-11 | 10 | yes |
| (L2) residual k-means codebook (unsupervised) | 2 B | 0.891 | (features) 0.842 | +0.049 ± 0.005 | 6.26e-09 | 1.25e-08 | 10 | yes |
| (L3) local classifier, quantised top posteriors | 2 B | 0.970 | (features) 0.842 | +0.128 ± 0.006 | 4.01e-12 | 3.61e-11 | 10 | yes |
| (L1) end-to-end encoder + centre, binary bottleneck | 4 B | 0.967 | (features) 0.906 | +0.061 ± 0.003 | 5.59e-12 | 3.91e-11 | 10 | yes |
| (L2) residual k-means codebook (unsupervised) | 4 B | 0.936 | (features) 0.906 | +0.030 ± 0.003 | 2.36e-09 | 7.07e-09 | 10 | yes |
| (L3) local classifier, quantised top posteriors | 4 B | 0.970 | (features) 0.906 | +0.064 ± 0.003 | 4.07e-12 | 3.61e-11 | 10 | yes |

## Macro-F1 by budget (17 faults, calibrated 5% threshold)

| Encoding | 1 B | 2 B | 4 B | 8 B |
|---|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | 0.810 ± 0.005 | 0.810 ± 0.005 | 0.810 ± 0.005 | 0.810 ± 0.005 |
| (b) quantised summary stats | 0.489 ± 0.007 | 0.596 ± 0.002 | 0.724 ± 0.006 | 0.923 ± 0.004 |
| (c) quantised hand features (PCA/FFT) | 0.823 ± 0.008 | 0.842 ± 0.007 | 0.905 ± 0.004 | 0.947 ± 0.003 |
| (L1) end-to-end encoder + centre, binary bottleneck | 0.964 ± 0.003 | 0.965 ± 0.003 | 0.967 ± 0.003 | 0.966 ± 0.003 |
| (L2) residual k-means codebook (unsupervised) | 0.832 ± 0.007 | 0.890 ± 0.007 | 0.936 ± 0.003 | 0.946 ± 0.003 |
| (L3) local classifier, quantised top posteriors | 0.969 ± 0.002 | 0.970 ± 0.002 | 0.970 ± 0.003 | 0.971 ± 0.002 |
| (e) oracle (all raw variables, no byte limit) | 0.952 ± 0.003 |  |  |  |

## Realised test false-alarm rate (target 5%)

| Encoding | 1 B | 2 B | 4 B | 8 B |
|---|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | 0.045 ± 0.005 | 0.045 ± 0.005 | 0.045 ± 0.005 | 0.045 ± 0.005 |
| (b) quantised summary stats | 0.048 ± 0.005 | 0.049 ± 0.004 | 0.049 ± 0.005 | 0.048 ± 0.004 |
| (c) quantised hand features (PCA/FFT) | 0.050 ± 0.006 | 0.049 ± 0.005 | 0.049 ± 0.004 | 0.050 ± 0.005 |
| (L1) end-to-end encoder + centre, binary bottleneck | 0.048 ± 0.001 | 0.050 ± 0.002 | 0.051 ± 0.002 | 0.050 ± 0.002 |
| (L2) residual k-means codebook (unsupervised) | 0.045 ± 0.002 | 0.047 ± 0.003 | 0.052 ± 0.003 | 0.053 ± 0.004 |
| (L3) local classifier, quantised top posteriors | 0.054 ± 0.003 | 0.054 ± 0.002 | 0.053 ± 0.003 | 0.052 ± 0.002 |
| (e) oracle (all raw variables, no byte limit) | 0.050 ± 0.004 |  |  |  |

## Detection delay in samples (mean over detected fault runs)

| Encoding | 1 B | 2 B | 4 B | 8 B |
|---|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | 42.3 ± 2.1 | 42.3 ± 2.1 | 42.3 ± 2.1 | 42.3 ± 2.1 |
| (b) quantised summary stats | 97.4 ± 5.1 | 75.9 ± 4.3 | 47.6 ± 2.9 | 27.4 ± 0.7 |
| (c) quantised hand features (PCA/FFT) | 38.2 ± 1.3 | 38.4 ± 1.7 | 35.9 ± 0.9 | 29.7 ± 0.7 |
| (L1) end-to-end encoder + centre, binary bottleneck | 27.0 ± 0.5 | 25.8 ± 0.5 | 26.2 ± 0.4 | 27.5 ± 0.5 |
| (L2) residual k-means codebook (unsupervised) | 37.8 ± 0.6 | 36.8 ± 0.5 | 32.8 ± 0.4 | 31.9 ± 0.6 |
| (L3) local classifier, quantised top posteriors | 27.8 ± 0.5 | 27.3 ± 0.5 | 27.2 ± 0.5 | 26.1 ± 0.5 |
| (e) oracle (all raw variables, no byte limit) | 27.3 ± 0.6 |  |  |  |

## Macro-F1 re-thresholded to exactly 5% test false alarms

| Encoding | 1 B | 2 B | 4 B | 8 B |
|---|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | 0.812 ± 0.007 | 0.812 ± 0.007 | 0.812 ± 0.007 | 0.812 ± 0.007 |
| (b) quantised summary stats | 0.490 ± 0.007 | 0.596 ± 0.002 | 0.724 ± 0.006 | 0.922 ± 0.004 |
| (c) quantised hand features (PCA/FFT) | 0.823 ± 0.008 | 0.842 ± 0.007 | 0.906 ± 0.004 | 0.947 ± 0.003 |
| (L1) end-to-end encoder + centre, binary bottleneck | 0.963 ± 0.003 | 0.965 ± 0.003 | 0.967 ± 0.003 | 0.966 ± 0.003 |
| (L2) residual k-means codebook (unsupervised) | 0.833 ± 0.007 | 0.891 ± 0.007 | 0.936 ± 0.003 | 0.946 ± 0.003 |
| (L3) local classifier, quantised top posteriors | 0.970 ± 0.003 | 0.970 ± 0.002 | 0.970 ± 0.003 | 0.972 ± 0.002 |
| (e) oracle (all raw variables, no byte limit) | 0.952 ± 0.003 |  |  |  |

## Ablations of the best learned method

### (L3) local classifier, quantised top posteriors at 1 B

The centre is trained on intact messages; the intervention only changes what it receives at test time. The method shown is the one with the best test macro-F1 at this budget (chosen after the fact, for description only).

| Condition | Macro-F1 | Drop from intact | Test false alarms | Detection delay |
|---|---|---|---|---|
| intact | 0.969 ± 0.002 | - | 0.054 ± 0.003 | 27.8 ± 0.5 |
| node zeroed: reactor | 0.911 ± 0.011 | +0.059 ± 0.011 | 0.076 ± 0.032 | 29.9 ± 1.8 |
| node zeroed: separator | 0.961 ± 0.009 | +0.008 ± 0.007 | 0.065 ± 0.029 | 28.1 ± 1.3 |
| node zeroed: stripper | 0.929 ± 0.007 | +0.040 ± 0.005 | 0.085 ± 0.054 | 30.6 ± 3.5 |
| node zeroed: compressor_utilities | 0.673 ± 0.027 | +0.297 ± 0.027 | 0.038 ± 0.018 | 60.4 ± 4.1 |
| messages shuffled across runs | 0.053 ± 0.008 | +0.916 ± 0.009 | 0.267 ± 0.008 | 36.8 ± 1.8 |

### (L3) local classifier, quantised top posteriors at 2 B

The centre is trained on intact messages; the intervention only changes what it receives at test time. The method shown is the one with the best test macro-F1 at this budget (chosen after the fact, for description only).

| Condition | Macro-F1 | Drop from intact | Test false alarms | Detection delay |
|---|---|---|---|---|
| intact | 0.970 ± 0.002 | - | 0.054 ± 0.002 | 27.3 ± 0.5 |
| node zeroed: reactor | 0.909 ± 0.010 | +0.061 ± 0.010 | 0.151 ± 0.070 | 26.2 ± 1.9 |
| node zeroed: separator | 0.959 ± 0.006 | +0.011 ± 0.006 | 0.086 ± 0.040 | 27.2 ± 1.5 |
| node zeroed: stripper | 0.928 ± 0.005 | +0.042 ± 0.004 | 0.073 ± 0.029 | 33.9 ± 4.6 |
| node zeroed: compressor_utilities | 0.701 ± 0.025 | +0.269 ± 0.024 | 0.100 ± 0.044 | 44.3 ± 6.1 |
| messages shuffled across runs | 0.053 ± 0.008 | +0.917 ± 0.009 | 0.266 ± 0.007 | 37.1 ± 2.4 |

### (L3) local classifier, quantised top posteriors at 4 B

The centre is trained on intact messages; the intervention only changes what it receives at test time. The method shown is the one with the best test macro-F1 at this budget (chosen after the fact, for description only).

| Condition | Macro-F1 | Drop from intact | Test false alarms | Detection delay |
|---|---|---|---|---|
| intact | 0.970 ± 0.003 | - | 0.053 ± 0.003 | 27.2 ± 0.5 |
| node zeroed: reactor | 0.899 ± 0.012 | +0.070 ± 0.012 | 0.235 ± 0.094 | 25.0 ± 2.9 |
| node zeroed: separator | 0.948 ± 0.013 | +0.021 ± 0.013 | 0.157 ± 0.073 | 25.7 ± 2.1 |
| node zeroed: stripper | 0.920 ± 0.004 | +0.050 ± 0.004 | 0.061 ± 0.039 | 38.3 ± 4.4 |
| node zeroed: compressor_utilities | 0.716 ± 0.022 | +0.254 ± 0.022 | 0.277 ± 0.115 | 25.0 ± 5.4 |
| messages shuffled across runs | 0.053 ± 0.008 | +0.916 ± 0.009 | 0.264 ± 0.009 | 36.6 ± 1.3 |

## Chosen hyperparameters (tuning block)

| Method | Budget | Hyperparameters | Best tuning-block validation macro-F1 |
|---|---|---|---|
| e2e | 1 B | {'bottleneck': 'ste', 'hidden': 64, 'epochs': 20, 'lr': 0.002} | 0.965 |
| e2e | 2 B | {'bottleneck': 'gumbel', 'hidden': 64, 'epochs': 20, 'lr': 0.002} | 0.968 |
| e2e | 4 B | {'bottleneck': 'gumbel', 'hidden': 64, 'epochs': 20, 'lr': 0.002} | 0.968 |
| e2e | 8 B | {'bottleneck': 'ste', 'hidden': 64, 'epochs': 20, 'lr': 0.002} | 0.968 |
| local | 1 B | {'hidden': 32, 'l2': 0.001, 'max_iter': 200} | 0.965 |
| local | 2 B | {'hidden': 32, 'l2': 0.001, 'max_iter': 100} | 0.967 |
| local | 4 B | {'hidden': 32, 'l2': 0.001, 'max_iter': 60} | 0.966 |
| local | 8 B | {'hidden': 32, 'l2': 0.001, 'max_iter': 100} | 0.967 |
| vq | 1 B | {'whiten': False, 'dim': 8} | 0.832 |
| vq | 2 B | {'whiten': False, 'dim': 8} | 0.895 |
| vq | 4 B | {'whiten': False, 'dim': 32} | 0.931 |
| vq | 8 B | {'whiten': False, 'dim': 32} | 0.943 |

## Checks that ran with the experiment

- Largest message: 8 B per node at most (rows with a message longer than their budget: 0); largest symbol 255 (a byte holds 0..255).

## Caveats

- One simulator, one sampling interval; replicates differ in simulator seeds only. The hand-design scores are step 2's numbers on the same replicates, not re-run.
- The centre's settings come from step 2's search with hand features; a search with the learned codes (a different input distribution) might help them a little. The encoders got a staged coordinate search of at most about 10 fits per (method, budget) on the tuning block; the hand designs are fixed designs with step 2's centre search.
- The learned codes (L1, L3) use the fault labels of the training runs to shape the message; the hand designs (a)-(c) are unsupervised and only their centre sees labels. A win therefore says that a label-trained node code beats these particular unsupervised designs, not that no hand-designed supervised code (for example a per-node discriminant) could match it.
- Learned encoders are trained on 20 faulty runs per fault (340 faulty runs plus 40 fault-free, in all), the same data the hand designs' PCA and thresholds used.
- Windows within a run are autocorrelated; replicates, not windows, are the unit of the CIs.
- The tuning block is disjoint from every replicate; the best-method choice in the ablation uses test scores.

## Reproduce

```bash
python scripts/plant_learned_codes.py --methods e2e,vq,local --budgets 1,2,4,8 --replicates 10 --workers 8 --out docs/results/plant_learned_codes.md
# merge partial result files into the report, no computation:
python scripts/plant_learned_codes.py --from-json <dir or files> --out docs/results/plant_learned_codes.md
```
