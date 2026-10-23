# Plant floor: diagnosis quality vs uplink bytes, run-level measuring stick (R8, step 2)

- Date: 2026-10-23; on top of base commit 1bcf1f1; Python 3.13.7; Intel64 Family 6 Model 183 Stepping 1, GenuineIntel; 4 worker processes; 32.0 min wall clock (126 CPU-minutes of tuning and cell work)
- Data: Additional Tennessee Eastman Process Simulation Data for Anomaly Detection Evaluation (Rieth et al. 2017) (https://doi.org/10.7910/DVN/6C3JR1, retrieved 2026-10-23). 500 independent simulation runs for each of the fault-free condition and faults 1..20 in each of four files (fault-free/faulty x training/testing); training runs have 500 samples (25 h), testing runs 960 (48 h), one sample every 3 minutes, the same 52 variables as the Braatz data. Measured from the data: the first faulty sample (0-based) is 20 in the training runs and 160 in the testing runs, i.e. the fault is introduced after 1 h and 8 h. Fault 21 of the Braatz set is not in this data. Licence: Public domain dedication (Harvard Dataverse terms of use for this dataset): the contributors dedicated the work to the public domain and waived their rights worldwide under copyright law; it may be copied, modified, distributed and performed for any lawful purpose without asking permission. No warranty and no implied endorsement by Pacific Science & Engineering Group, Inc. or the authors. The full download (1.4 GB) was used; no subset of the published runs. `python scripts/fetch_rieth.py` downloads it to a cache outside the repo, checks size, MD5 and SHA-256 of every file and converts it once to float32 `.npy` arrays; the conversion was cross-checked against pyreadr.
- Run-level replicates: **10 independent replicates**, each with its own disjoint simulation runs. Per replicate and fault: 20 training runs (training files), 12 validation and 25 test runs (testing files); fault-free: 40 training, 20 validation, 25 test runs. A replicate's test set is 450 runs; the 10 replicates together use 250 of the 500 testing runs of every fault. A separate tuning block (also disjoint) is used for hyperparameters. Every number is mean ± 95% t-CI over replicates (`evaluation.stats.mean_ci`), so the CIs cover the draw of training, calibration and test runs as well as classifier initialisation. A dagger (†) marks a cell measured on the first 5 replicates only, to keep the run short: the random-bytes control at 8 and 51 B, the centre ablation and the 20-fault set.
- Protocol: window 20 samples (60 min), one uplink and one diagnosis every 10 samples (30 min); test runs have the fault from sample 160 (95 windows per run, 15 of them before the onset); the centre never sees the clock or the onset. Window label = the run's fault once the window contains a faulty sample (>= 1 faulty sample), else normal. Training windows use the same stride.
- Centre: an MLP (one tanh hidden layer, class-balanced, L2, L-BFGS) on the decoded messages, optionally stacking the last few uplinks; hyperparameters in the tuning section. Calibration: the normal margin `tau` is the 95% quantile of the margin on the validation normal windows (validation fault-free runs plus the pre-onset windows of validation fault runs, the same mix as the test normal windows), so 5% of them alarm; the same target for every encoding, nothing of the validation set is trained on.
- LoRaWAN EU868 reference: 51 B at SF12 is 2.79 s on air, so a 1% duty cycle allows one uplink per 279 s per node; our uplinks are 1800 s apart, so every budget here is legal at SF12.

## Variable mapping (52 TEP variables to 4 nodes)

Each node owns the sensors and actuators of one unit operation or service; analysers are attached to the unit that the sampled stream leaves.

- **reactor** (18 variables): XMEAS(1) A feed (stream 1) [kscmh]; XMEAS(2) D feed (stream 2) [kg/h]; XMEAS(3) E feed (stream 3) [kg/h]; XMEAS(4) A and C feed (stream 4) [kscmh]; XMEAS(6) Reactor feed rate (stream 6) [kscmh]; XMEAS(7) Reactor pressure [kPa gauge]; XMEAS(8) Reactor level [%]; XMEAS(9) Reactor temperature [deg C]; XMEAS(23) Reactor feed analysis, component A (stream 6) [mol%]; XMEAS(24) Reactor feed analysis, component B (stream 6) [mol%]; XMEAS(25) Reactor feed analysis, component C (stream 6) [mol%]; XMEAS(26) Reactor feed analysis, component D (stream 6) [mol%]; XMEAS(27) Reactor feed analysis, component E (stream 6) [mol%]; XMEAS(28) Reactor feed analysis, component F (stream 6) [mol%]; XMV(1) D feed flow (stream 2) [%]; XMV(2) E feed flow (stream 3) [%]; XMV(3) A feed flow (stream 1) [%]; XMV(4) A and C feed flow (stream 4) [%]
- **separator** (15 variables): XMEAS(10) Purge rate (stream 9) [kscmh]; XMEAS(11) Product separator temperature [deg C]; XMEAS(12) Product separator level [%]; XMEAS(13) Product separator pressure [kPa gauge]; XMEAS(14) Product separator underflow (stream 10) [m3/h]; XMEAS(29) Purge gas analysis, component A (stream 9) [mol%]; XMEAS(30) Purge gas analysis, component B (stream 9) [mol%]; XMEAS(31) Purge gas analysis, component C (stream 9) [mol%]; XMEAS(32) Purge gas analysis, component D (stream 9) [mol%]; XMEAS(33) Purge gas analysis, component E (stream 9) [mol%]; XMEAS(34) Purge gas analysis, component F (stream 9) [mol%]; XMEAS(35) Purge gas analysis, component G (stream 9) [mol%]; XMEAS(36) Purge gas analysis, component H (stream 9) [mol%]; XMV(6) Purge valve (stream 9) [%]; XMV(7) Separator pot liquid flow (stream 10) [%]
- **stripper** (10 variables): XMEAS(15) Stripper level [%]; XMEAS(16) Stripper pressure [kPa gauge]; XMEAS(17) Stripper underflow (stream 11) [m3/h]; XMEAS(18) Stripper temperature [deg C]; XMEAS(37) Product analysis, component D (stream 11) [mol%]; XMEAS(38) Product analysis, component E (stream 11) [mol%]; XMEAS(39) Product analysis, component F (stream 11) [mol%]; XMEAS(40) Product analysis, component G (stream 11) [mol%]; XMEAS(41) Product analysis, component H (stream 11) [mol%]; XMV(8) Stripper liquid product flow (stream 11) [%]
- **compressor_utilities** (9 variables): XMEAS(5) Recycle flow (stream 8) [kscmh]; XMEAS(19) Stripper steam flow [kg/h]; XMEAS(20) Compressor work [kW]; XMEAS(21) Reactor cooling water outlet temperature [deg C]; XMEAS(22) Separator cooling water outlet temperature [deg C]; XMV(5) Compressor recycle valve [%]; XMV(9) Stripper steam valve [%]; XMV(10) Reactor cooling water flow [%]; XMV(11) Condenser cooling water flow [%]

## Summary (17 faults)

- **The bar for learned codes** (best hand design per budget, centre calibrated on validation runs to 5% false alarms): 4 B: (features) macro-F1 0.905, test false alarms 4.9%, delay 35.9; 8 B: (features) macro-F1 0.947, test false alarms 5.0%, delay 29.7; 16 B: (summary) macro-F1 0.954, test false alarms 5.3%, delay 28.3; 51 B: (summary) macro-F1 0.955, test false alarms 5.2%, delay 28.1.
- Oracle (all raw variables, no byte limit): macro-F1 0.952, test false alarms 5.0%; silent 0.000; random bytes 0.004 (16 B).
- Realised test false-alarm rate against the 5% target: from 4.5% to 5.3% over the cells (mean 5.0%); the table below splits it by source.
- Headroom (oracle minus the best hand design, paired by replicate, macro-F1): 1 B +0.130 ± 0.007, 2 B +0.111 ± 0.006, 4 B +0.047 ± 0.005, 8 B +0.005 ± 0.001, 16 B -0.002 ± 0.002, 51 B -0.003 ± 0.002. The best hand design is statistically indistinguishable from the oracle at 16 B, 51 B; there is measurable room for a better code at 1 B, 2 B, 4 B, 8 B.
- Macro-F1 of (c) by budget: 1 B 0.823, 2 B 0.842, 4 B 0.905, 8 B 0.947, 16 B 0.951, 51 B 0.948.
- Macro-F1 of (b) by budget: 1 B 0.489, 2 B 0.596, 4 B 0.724, 8 B 0.923, 16 B 0.954, 51 B 0.955; the fixed one-byte alarm code (a): 0.810.

## Summary (20 faults)

- **The bar for learned codes** (best hand design per budget, centre calibrated on validation runs to 5% false alarms): 16 B: (summary) macro-F1 0.827, test false alarms 4.7%, delay 52.4.
- Oracle (all raw variables, no byte limit): macro-F1 0.812, test false alarms 5.1%; silent 0.000.
- Realised test false-alarm rate against the 5% target: from 4.3% to 5.1% over the cells (mean 4.8%); the table below splits it by source.

## Oracle sanity check

The oracle reads unquantised statistics of all 52 variables plus every PCA and band feature (c) can send, with no byte limit, and uses the same centre family and the same tuning effort as the encodings, so it should do at least as well as every encoding. Paired by replicate (same runs), oracle minus encoding; 'within the CI' means the mean is below zero but the 95% interval reaches it.

**Macro-F1 at the calibrated 5% threshold (17 faults)**

| Encoding | Budget | Oracle - encoding | Oracle at least as good? |
|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | 1 B | +0.142 ± 0.005 | yes |
| (b) quantised summary stats (mean, std, max, min) | 1 B | +0.463 ± 0.007 | yes |
| (b) quantised summary stats (mean, std, max, min) | 2 B | +0.356 ± 0.003 | yes |
| (b) quantised summary stats (mean, std, max, min) | 4 B | +0.229 ± 0.005 | yes |
| (b) quantised summary stats (mean, std, max, min) | 8 B | +0.030 ± 0.003 | yes |
| (b) quantised summary stats (mean, std, max, min) | 16 B | -0.002 ± 0.002 | yes, within the CI |
| (b) quantised summary stats (mean, std, max, min) | 51 B | -0.003 ± 0.002 | NO |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | 1 B | +0.130 ± 0.007 | yes |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | 2 B | +0.111 ± 0.006 | yes |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | 4 B | +0.047 ± 0.005 | yes |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | 8 B | +0.005 ± 0.001 | yes |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | 16 B | +0.001 ± 0.001 | yes |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | 51 B | +0.004 ± 0.002 | yes |
| control: random bytes | 8 B | +0.946 ± 0.006 | yes |
| control: random bytes | 16 B | +0.948 ± 0.003 | yes |
| control: random bytes | 51 B | +0.947 ± 0.005 | yes |

Result: the oracle is significantly below summary 51 B.

**Macro-F1 re-thresholded to exactly 5% test false alarms (17 faults)**

| Encoding | Budget | Oracle - encoding | Oracle at least as good? |
|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | 1 B | +0.140 ± 0.006 | yes |
| (b) quantised summary stats (mean, std, max, min) | 1 B | +0.463 ± 0.007 | yes |
| (b) quantised summary stats (mean, std, max, min) | 2 B | +0.356 ± 0.003 | yes |
| (b) quantised summary stats (mean, std, max, min) | 4 B | +0.228 ± 0.005 | yes |
| (b) quantised summary stats (mean, std, max, min) | 8 B | +0.030 ± 0.003 | yes |
| (b) quantised summary stats (mean, std, max, min) | 16 B | -0.002 ± 0.002 | yes, within the CI |
| (b) quantised summary stats (mean, std, max, min) | 51 B | -0.003 ± 0.002 | NO |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | 1 B | +0.130 ± 0.007 | yes |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | 2 B | +0.111 ± 0.006 | yes |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | 4 B | +0.047 ± 0.005 | yes |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | 8 B | +0.005 ± 0.002 | yes |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | 16 B | +0.001 ± 0.002 | yes |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | 51 B | +0.004 ± 0.002 | yes |
| control: random bytes | 8 B | +0.947 ± 0.006 | yes |
| control: random bytes | 16 B | +0.948 ± 0.003 | yes |
| control: random bytes | 51 B | +0.948 ± 0.005 | yes |

Result: the oracle is significantly below summary 51 B.

## Results: 17 diagnosable faults (faults 3, 9 and 15 left out: no data-driven method separates them from normal operation, see the 20-fault section)

### Macro-F1 over the fault classes at the calibrated 5% threshold (higher is better)

| Encoding | 0 B | 1 B | 2 B | 4 B | 8 B | 16 B | 51 B |
|---|---|---|---|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | - | 0.810 ± 0.005 | 0.810 ± 0.005 | 0.810 ± 0.005 | 0.810 ± 0.005 | 0.810 ± 0.005 | 0.810 ± 0.005 |
| (b) quantised summary stats (mean, std, max, min) | - | 0.489 ± 0.007 | 0.596 ± 0.002 | 0.724 ± 0.006 | 0.923 ± 0.004 | 0.954 ± 0.003 | 0.955 ± 0.003 |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | - | 0.823 ± 0.008 | 0.842 ± 0.007 | 0.905 ± 0.004 | 0.947 ± 0.003 | 0.951 ± 0.003 | 0.948 ± 0.004 |
| (d) silent (0 B, always reports normal) | 0.000 ± 0.000 | - | - | - | - | - | - |
| (e) oracle (all raw variables, no byte limit) | 0.952 ± 0.003 | - | - | - | - | - | - |
| control: random bytes | - | - | - | - | 0.004 ± 0.000† | 0.004 ± 0.001 | 0.003 ± 0.001† |

Mean ± 95% CI over 10 replicates. F1 of each of the 17 fault classes from the confusion of all test windows (normal windows mistaken for a fault count against that class), then averaged. Random-guess level is about 0.05. The alarm byte fits every budget, so (a) is one measurement shown in each column.

### False-alarm rate on the test normal windows (target 5%)

| Encoding | 0 B | 1 B | 2 B | 4 B | 8 B | 16 B | 51 B |
|---|---|---|---|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | - | 0.045 ± 0.005 | 0.045 ± 0.005 | 0.045 ± 0.005 | 0.045 ± 0.005 | 0.045 ± 0.005 | 0.045 ± 0.005 |
| (b) quantised summary stats (mean, std, max, min) | - | 0.048 ± 0.005 | 0.049 ± 0.004 | 0.049 ± 0.005 | 0.048 ± 0.004 | 0.053 ± 0.004 | 0.052 ± 0.004 |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | - | 0.050 ± 0.006 | 0.049 ± 0.005 | 0.049 ± 0.004 | 0.050 ± 0.005 | 0.051 ± 0.005 | 0.050 ± 0.005 |
| (d) silent (0 B, always reports normal) | 0.000 ± 0.000 | - | - | - | - | - | - |
| (e) oracle (all raw variables, no byte limit) | 0.050 ± 0.004 | - | - | - | - | - | - |
| control: random bytes | - | - | - | - | 0.053 ± 0.005† | 0.052 ± 0.003 | 0.053 ± 0.005† |

Realised share of test normal windows (fault-free runs plus the pre-onset part of every fault run) that raise an alarm, with `tau` calibrated on validation runs to 5%.

### Detection delay in samples (lower is better; 1 sample = 3 min)

| Encoding | 0 B | 1 B | 2 B | 4 B | 8 B | 16 B | 51 B |
|---|---|---|---|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | - | 42.3 ± 2.1 | 42.3 ± 2.1 | 42.3 ± 2.1 | 42.3 ± 2.1 | 42.3 ± 2.1 | 42.3 ± 2.1 |
| (b) quantised summary stats (mean, std, max, min) | - | 97.4 ± 5.1 | 75.9 ± 4.3 | 47.6 ± 2.9 | 27.4 ± 0.7 | 28.3 ± 0.3 | 28.1 ± 0.8 |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | - | 38.2 ± 1.3 | 38.4 ± 1.7 | 35.9 ± 0.9 | 29.7 ± 0.7 | 29.9 ± 0.6 | 30.0 ± 0.5 |
| (d) silent (0 B, always reports normal) | never | - | - | - | - | - | - |
| (e) oracle (all raw variables, no byte limit) | 27.3 ± 0.6 | - | - | - | - | - | - |
| control: random bytes | - | - | - | - | 219.6 ± 10.8† | 225.5 ± 8.7 | 261.4 ± 22.3† |

Samples from the fault onset to the end of the first window in which the centre raises any alarm, averaged over the detected fault runs (the minimum is 10, one stride).

### Detection rate (higher is better)

| Encoding | 0 B | 1 B | 2 B | 4 B | 8 B | 16 B | 51 B |
|---|---|---|---|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | - | 1.000 ± 0.001 | 1.000 ± 0.001 | 1.000 ± 0.001 | 1.000 ± 0.001 | 1.000 ± 0.001 | 1.000 ± 0.001 |
| (b) quantised summary stats (mean, std, max, min) | - | 0.984 ± 0.005 | 0.987 ± 0.005 | 0.999 ± 0.002 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | - | 1.000 ± 0.001 | 1.000 ± 0.001 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| (d) silent (0 B, always reports normal) | 0.000 ± 0.000 | - | - | - | - | - | - |
| (e) oracle (all raw variables, no byte limit) | 1.000 ± 0.000 | - | - | - | - | - | - |
| control: random bytes | - | - | - | - | 0.958 ± 0.030† | 0.950 ± 0.007 | 0.896 ± 0.032† |

Share of the fault runs in which the centre ever alarmed after the onset.

### The same decisions re-thresholded to exactly 5% false alarms on the test normal windows

Each centre's threshold is moved so that exactly 5% of its test normal windows alarm. This looks at test labels, so it is not a deployable setting; it removes the residual calibration error so that encodings compare at one operating point. The silent row is unchanged.

**Macro-F1 (higher is better)**

| Encoding | 0 B | 1 B | 2 B | 4 B | 8 B | 16 B | 51 B |
|---|---|---|---|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | - | 0.812 ± 0.007 | 0.812 ± 0.007 | 0.812 ± 0.007 | 0.812 ± 0.007 | 0.812 ± 0.007 | 0.812 ± 0.007 |
| (b) quantised summary stats (mean, std, max, min) | - | 0.490 ± 0.007 | 0.596 ± 0.002 | 0.724 ± 0.006 | 0.922 ± 0.004 | 0.954 ± 0.003 | 0.956 ± 0.003 |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | - | 0.823 ± 0.008 | 0.842 ± 0.007 | 0.906 ± 0.004 | 0.947 ± 0.003 | 0.951 ± 0.003 | 0.948 ± 0.004 |
| (d) silent (0 B, always reports normal) | 0.000 ± 0.000 | - | - | - | - | - | - |
| (e) oracle (all raw variables, no byte limit) | 0.952 ± 0.003 | - | - | - | - | - | - |
| control: random bytes | - | - | - | - | 0.004 ± 0.000† | 0.004 ± 0.000 | 0.003 ± 0.000† |

Mean ± 95% CI over 10 replicates.

**Detection delay in samples**

| Encoding | 0 B | 1 B | 2 B | 4 B | 8 B | 16 B | 51 B |
|---|---|---|---|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | - | 39.6 ± 2.1 | 39.6 ± 2.1 | 39.6 ± 2.1 | 39.6 ± 2.1 | 39.6 ± 2.1 | 39.6 ± 2.1 |
| (b) quantised summary stats (mean, std, max, min) | - | 95.8 ± 4.0 | 74.2 ± 4.6 | 46.8 ± 1.7 | 27.2 ± 0.6 | 28.5 ± 0.3 | 28.2 ± 0.9 |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | - | 38.4 ± 1.9 | 38.6 ± 1.8 | 35.7 ± 0.8 | 29.7 ± 0.7 | 29.9 ± 0.6 | 30.0 ± 0.7 |
| (d) silent (0 B, always reports normal) | never | - | - | - | - | - | - |
| (e) oracle (all raw variables, no byte limit) | 27.3 ± 0.5 | - | - | - | - | - | - |
| control: random bytes | - | - | - | - | 227.3 ± 10.1† | 232.3 ± 8.8 | 273.8 ± 20.4† |

Mean over the detected fault runs.

### Where the realised false alarms come from (tuned cells)

| Cell | Validation (at `tau`) | Test, all normal windows | Test, fault-free runs | Test, pre-onset windows |
|---|---|---|---|---|
| (a) threshold alarms 1 B | 0.050 ± 0.000 | 0.045 ± 0.005 | 0.072 ± 0.008 | 0.035 ± 0.005 |
| (b) quantised summary stats 16 B | 0.050 ± 0.000 | 0.053 ± 0.004 | 0.042 ± 0.004 | 0.057 ± 0.005 |
| (c) quantised hand features 16 B | 0.050 ± 0.000 | 0.051 ± 0.005 | 0.027 ± 0.005 | 0.060 ± 0.006 |
| control: random bytes 16 B | 0.050 ± 0.000 | 0.052 ± 0.003 | 0.042 ± 0.003 | 0.056 ± 0.004 |
| (e) oracle | 0.050 ± 0.000 | 0.050 ± 0.004 | 0.040 ± 0.005 | 0.053 ± 0.005 |

Validation runs are fault-free runs (960 samples) plus the first 160 samples of fault runs, exactly like the test normal windows, so a gap between the columns is the run-to-run difference between independent draws of runs.

### Mean bytes actually used per node message

| Encoding | 0 B | 1 B | 2 B | 4 B | 8 B | 16 B | 51 B |
|---|---|---|---|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | - | 1.0 | - | - | - | - | - |
| (b) quantised summary stats (mean, std, max, min) | - | 1.0 | 2.0 | 4.0 | 8.0 | 16.0 | 35.5 |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | - | 1.0 | 2.0 | 4.0 | 8.0 | 16.0 | 31.8 |
| (d) silent (0 B, always reports normal) | 0.0 | - | - | - | - | - | - |
| (e) oracle (all raw variables, no byte limit) | 0.0 | - | - | - | - | - | - |
| control: random bytes | - | - | - | - | 8.0 | 16.0 | 51.0 |

## Results: all 20 faults

### Macro-F1 over the fault classes at the calibrated 5% threshold (higher is better)

| Encoding | 0 B | 1 B | 2 B | 4 B | 8 B | 16 B | 51 B |
|---|---|---|---|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | - | 0.683 ± 0.010† | 0.683 ± 0.010† | 0.683 ± 0.010† | 0.683 ± 0.010† | 0.683 ± 0.010† | 0.683 ± 0.010† |
| (b) quantised summary stats (mean, std, max, min) | - | - | - | - | - | 0.827 ± 0.008† | - |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | - | - | - | - | - | 0.806 ± 0.006† | - |
| (d) silent (0 B, always reports normal) | 0.000 ± 0.000† | - | - | - | - | - | - |
| (e) oracle (all raw variables, no byte limit) | 0.812 ± 0.006† | - | - | - | - | - | - |
| control: random bytes | - | - | - | - | - | - | - |

Mean ± 95% CI over 10 replicates. F1 of each of the 20 fault classes from the confusion of all test windows (normal windows mistaken for a fault count against that class), then averaged. Random-guess level is about 0.05. The alarm byte fits every budget, so (a) is one measurement shown in each column.

### False-alarm rate on the test normal windows (target 5%)

| Encoding | 0 B | 1 B | 2 B | 4 B | 8 B | 16 B | 51 B |
|---|---|---|---|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | - | 0.043 ± 0.012† | 0.043 ± 0.012† | 0.043 ± 0.012† | 0.043 ± 0.012† | 0.043 ± 0.012† | 0.043 ± 0.012† |
| (b) quantised summary stats (mean, std, max, min) | - | - | - | - | - | 0.047 ± 0.007† | - |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | - | - | - | - | - | 0.050 ± 0.006† | - |
| (d) silent (0 B, always reports normal) | 0.000 ± 0.000† | - | - | - | - | - | - |
| (e) oracle (all raw variables, no byte limit) | 0.051 ± 0.006† | - | - | - | - | - | - |
| control: random bytes | - | - | - | - | - | - | - |

Realised share of test normal windows (fault-free runs plus the pre-onset part of every fault run) that raise an alarm, with `tau` calibrated on validation runs to 5%.

### Detection delay in samples (lower is better; 1 sample = 3 min)

| Encoding | 0 B | 1 B | 2 B | 4 B | 8 B | 16 B | 51 B |
|---|---|---|---|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | - | 71.6 ± 5.9† | 71.6 ± 5.9† | 71.6 ± 5.9† | 71.6 ± 5.9† | 71.6 ± 5.9† | 71.6 ± 5.9† |
| (b) quantised summary stats (mean, std, max, min) | - | - | - | - | - | 52.4 ± 3.9† | - |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | - | - | - | - | - | 57.6 ± 3.2† | - |
| (d) silent (0 B, always reports normal) | never† | - | - | - | - | - | - |
| (e) oracle (all raw variables, no byte limit) | 51.2 ± 6.0† | - | - | - | - | - | - |
| control: random bytes | - | - | - | - | - | - | - |

Samples from the fault onset to the end of the first window in which the centre raises any alarm, averaged over the detected fault runs (the minimum is 10, one stride).

### Detection rate (higher is better)

| Encoding | 0 B | 1 B | 2 B | 4 B | 8 B | 16 B | 51 B |
|---|---|---|---|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | - | 0.978 ± 0.009† | 0.978 ± 0.009† | 0.978 ± 0.009† | 0.978 ± 0.009† | 0.978 ± 0.009† | 0.978 ± 0.009† |
| (b) quantised summary stats (mean, std, max, min) | - | - | - | - | - | 0.998 ± 0.002† | - |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | - | - | - | - | - | 0.993 ± 0.005† | - |
| (d) silent (0 B, always reports normal) | 0.000 ± 0.000† | - | - | - | - | - | - |
| (e) oracle (all raw variables, no byte limit) | 0.998 ± 0.003† | - | - | - | - | - | - |
| control: random bytes | - | - | - | - | - | - | - |

Share of the fault runs in which the centre ever alarmed after the onset.

### The same decisions re-thresholded to exactly 5% false alarms on the test normal windows

Each centre's threshold is moved so that exactly 5% of its test normal windows alarm. This looks at test labels, so it is not a deployable setting; it removes the residual calibration error so that encodings compare at one operating point. The silent row is unchanged.

**Macro-F1 (higher is better)**

| Encoding | 0 B | 1 B | 2 B | 4 B | 8 B | 16 B | 51 B |
|---|---|---|---|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | - | 0.685 ± 0.010† | 0.685 ± 0.010† | 0.685 ± 0.010† | 0.685 ± 0.010† | 0.685 ± 0.010† | 0.685 ± 0.010† |
| (b) quantised summary stats (mean, std, max, min) | - | - | - | - | - | 0.828 ± 0.007† | - |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | - | - | - | - | - | 0.806 ± 0.005† | - |
| (d) silent (0 B, always reports normal) | 0.000 ± 0.000† | - | - | - | - | - | - |
| (e) oracle (all raw variables, no byte limit) | 0.812 ± 0.005† | - | - | - | - | - | - |
| control: random bytes | - | - | - | - | - | - | - |

Mean ± 95% CI over 10 replicates.

**Detection delay in samples**

| Encoding | 0 B | 1 B | 2 B | 4 B | 8 B | 16 B | 51 B |
|---|---|---|---|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | - | 68.6 ± 7.3† | 68.6 ± 7.3† | 68.6 ± 7.3† | 68.6 ± 7.3† | 68.6 ± 7.3† | 68.6 ± 7.3† |
| (b) quantised summary stats (mean, std, max, min) | - | - | - | - | - | 51.6 ± 3.4† | - |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | - | - | - | - | - | 57.8 ± 3.3† | - |
| (d) silent (0 B, always reports normal) | never† | - | - | - | - | - | - |
| (e) oracle (all raw variables, no byte limit) | 52.3 ± 9.3† | - | - | - | - | - | - |
| control: random bytes | - | - | - | - | - | - | - |

Mean over the detected fault runs.

## What the stronger centre buys (centre ablation, 16 B for (c))

Macro-F1 at the calibrated 5% threshold with the same encoders, replicates and calibration; only the centre changes.

| Centre | (c) features 16 B | (e) oracle |
|---|---|---|
| tuned MLP with the selected context | 0.951 ± 0.003 | 0.952 ± 0.003 |
| softmax regression, no context (the step-1 centre) | 0.880 ± 0.008 | 0.915 ± 0.008 |
| tuned MLP without temporal context | 0.929 ± 0.008 | 0.932 ± 0.005 |

## Per-fault F1 (calibrated threshold, mean ± CI over replicates)

| Fault | alarms 1 B | summary 16 B | features 16 B | oracle |
|---|---|---|---|---|
| IDV(1) | 0.94 ± 0.01 | 0.99 ± 0.00 | 0.99 ± 0.00 | 0.99 ± 0.00 |
| IDV(2) | 0.98 ± 0.00 | 0.99 ± 0.00 | 0.99 ± 0.00 | 0.99 ± 0.00 |
| IDV(4) | 0.95 ± 0.00 | 0.98 ± 0.01 | 0.97 ± 0.01 | 0.99 ± 0.00 |
| IDV(5) | 0.96 ± 0.00 | 0.97 ± 0.01 | 0.93 ± 0.01 | 0.97 ± 0.01 |
| IDV(6) | 0.96 ± 0.01 | 0.99 ± 0.01 | 0.99 ± 0.00 | 0.99 ± 0.01 |
| IDV(7) | 0.92 ± 0.01 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| IDV(8) | 0.80 ± 0.01 | 0.94 ± 0.00 | 0.95 ± 0.00 | 0.95 ± 0.00 |
| IDV(10) | 0.69 ± 0.01 | 0.83 ± 0.01 | 0.83 ± 0.01 | 0.78 ± 0.01 |
| IDV(11) | 0.94 ± 0.00 | 0.98 ± 0.00 | 0.99 ± 0.00 | 0.98 ± 0.00 |
| IDV(12) | 0.66 ± 0.02 | 0.95 ± 0.02 | 0.93 ± 0.03 | 0.94 ± 0.02 |
| IDV(13) | 0.82 ± 0.01 | 0.88 ± 0.01 | 0.92 ± 0.01 | 0.91 ± 0.01 |
| IDV(14) | 0.33 ± 0.04 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| IDV(16) | 0.45 ± 0.04 | 0.92 ± 0.01 | 0.92 ± 0.01 | 0.90 ± 0.01 |
| IDV(17) | 0.97 ± 0.00 | 0.97 ± 0.00 | 0.97 ± 0.00 | 0.97 ± 0.00 |
| IDV(18) | 0.78 ± 0.03 | 0.88 ± 0.01 | 0.87 ± 0.02 | 0.89 ± 0.01 |
| IDV(19) | 0.66 ± 0.02 | 0.99 ± 0.00 | 0.99 ± 0.00 | 0.99 ± 0.00 |
| IDV(20) | 0.95 ± 0.00 | 0.95 ± 0.01 | 0.94 ± 0.01 | 0.95 ± 0.00 |

### All 20 faults: the faults that are not diagnosable

| Fault | features 16 B | oracle |
|---|---|---|
| IDV(1) | 0.99 ± 0.00† | 0.99 ± 0.00† |
| IDV(2) | 0.98 ± 0.00† | 0.99 ± 0.00† |
| IDV(3) | 0.03 ± 0.01† | 0.05 ± 0.01† |
| IDV(4) | 0.96 ± 0.02† | 0.98 ± 0.01† |
| IDV(5) | 0.88 ± 0.03† | 0.97 ± 0.01† |
| IDV(6) | 0.99 ± 0.00† | 1.00 ± 0.00† |
| IDV(7) | 0.96 ± 0.01† | 1.00 ± 0.01† |
| IDV(8) | 0.94 ± 0.01† | 0.94 ± 0.01† |
| IDV(9) | 0.03 ± 0.02† | 0.03 ± 0.01† |
| IDV(10) | 0.77 ± 0.04† | 0.73 ± 0.03† |
| IDV(11) | 0.99 ± 0.00† | 0.98 ± 0.00† |
| IDV(12) | 0.92 ± 0.05† | 0.93 ± 0.05† |
| IDV(13) | 0.92 ± 0.02† | 0.92 ± 0.02† |
| IDV(14) | 0.99 ± 0.00† | 1.00 ± 0.00† |
| IDV(15) | 0.09 ± 0.02† | 0.06 ± 0.01† |
| IDV(16) | 0.91 ± 0.01† | 0.88 ± 0.03† |
| IDV(17) | 0.97 ± 0.00† | 0.97 ± 0.00† |
| IDV(18) | 0.88 ± 0.03† | 0.90 ± 0.03† |
| IDV(19) | 0.98 ± 0.00† | 0.99 ± 0.00† |
| IDV(20) | 0.95 ± 0.01† | 0.96 ± 0.01† |

## Centre hyperparameters (chosen on validation runs only)

A staged search on the tuning block of the run splits (its own training runs; validation runs = its validation and test slices, none of which any replicate uses). One setting at a time, in the order below, every other setting at its current best; the score is the macro-F1 on the validation windows at the candidate's own calibrated threshold. Candidates within 0.002 of the best tie and the cheapest wins. Tuned once per encoding family at its reference budget (16 B; the alarm byte; the oracle) and reused at the other budgets; the random-bytes control uses (c)'s settings. Search space: context in [1, 2, 3, 4, 6]; hidden in [32, 64, 128]; l2 in [0.0001, 0.001, 0.01, 0.1]; max_iter in [60, 100, 200].

| Family | Reference cell | Selected | Fits | Validation macro-F1 before -> after |
|---|---|---|---|---|
| alarms | alarms 1 B | context=6, hidden=128, l2=0.001, max_iter=200 | 12 | 0.639 -> 0.815 |
| summary | summary 16 B | context=6, hidden=32, l2=0.001, max_iter=60 | 12 | 0.940 -> 0.953 |
| features | features 16 B | context=6, hidden=64, l2=0.001, max_iter=200 | 12 | 0.929 -> 0.951 |
| oracle | oracle | context=4, hidden=64, l2=0.0001, max_iter=60 | 12 | 0.941 -> 0.955 |

Full trace (stage, value, validation macro-F1):

- alarms: context=1 0.639; context=2 0.717; context=3 0.755; context=4 0.774; context=6 0.800; hidden=32 0.787; hidden=128 0.805; l2=0.0001 0.797; l2=0.01 0.778; l2=0.1 0.700; max_iter=60 0.791; max_iter=200 0.815
- summary: context=1 0.940; context=2 0.945; context=3 0.945; context=4 0.949; context=6 0.953; hidden=32 0.951; hidden=128 0.952; l2=0.0001 0.948; l2=0.01 0.952; l2=0.1 0.877; max_iter=60 0.951; max_iter=200 0.953
- features: context=1 0.929; context=2 0.936; context=3 0.942; context=4 0.947; context=6 0.949; hidden=32 0.946; hidden=128 0.949; l2=0.0001 0.945; l2=0.01 0.950; l2=0.1 0.884; max_iter=60 0.948; max_iter=200 0.951
- oracle: context=1 0.941; context=2 0.946; context=3 0.949; context=4 0.953; context=6 0.954; hidden=32 0.949; hidden=128 0.953; l2=0.0001 0.954; l2=0.01 0.952; l2=0.1 0.894; max_iter=60 0.955; max_iter=200 0.954

## Caveats

- All runs are simulations of one plant model with one sampling interval; the replicates differ in simulator random seeds only. The testing files hold 500 runs per condition, of which this report uses 370 per fault plus the tuning block.
- Windows within a run are strongly autocorrelated; replicates (not windows) are the unit of the CIs, which is why they are wide relative to the number of windows.
- Hyperparameters were tuned per encoding family at one budget and reused; (a)-(c) and the oracle got the same search, but a per-budget search might move the 8 B and 51 B cells slightly.
- The windows' labels follow the step-1 rule (a window is faulty once it contains one faulty sample), so the first windows after an onset are nearly unlabellable and cap macro-F1 below 1.
- Fault classes 3, 9 and 15 are not separable from normal operation here either; the 17-fault tables leave them out, the 20-fault tables show what they cost. Fault 21 of step 1 does not exist in this data, so numbers are not comparable one to one with step 1 (different data, fault set, centre and calibration).
- The alarm design is a fixed 1 B code, so its row does not change with the budget; the 0 B column of an encoding means no message, the same as (d).

## Fault list

- IDV(1): A/C feed ratio, B composition constant (stream 4), step
- IDV(2): B composition, A/C ratio constant (stream 4), step
- IDV(3): D feed temperature (stream 2), step
- IDV(4): Reactor cooling water inlet temperature, step
- IDV(5): Condenser cooling water inlet temperature, step
- IDV(6): A feed loss (stream 1), step
- IDV(7): C header pressure loss, reduced availability (stream 4), step
- IDV(8): A, B, C feed composition (stream 4), random variation
- IDV(9): D feed temperature (stream 2), random variation
- IDV(10): C feed temperature (stream 4), random variation
- IDV(11): Reactor cooling water inlet temperature, random variation
- IDV(12): Condenser cooling water inlet temperature, random variation
- IDV(13): Reaction kinetics, slow drift
- IDV(14): Reactor cooling water valve, sticking
- IDV(15): Condenser cooling water valve, sticking
- IDV(16): Unknown
- IDV(17): Unknown
- IDV(18): Unknown
- IDV(19): Unknown
- IDV(20): Unknown

## Reproduce

```bash
python scripts/fetch_rieth.py
python scripts/plant_baselines_rieth.py --replicates 10 --secondary 5 --workers 4 --out docs/results/plant_bytes_baselines_rieth.md
# re-render the report from the saved per-cell results, no computation:
python scripts/plant_baselines_rieth.py --from-json docs/results/plant_bytes_baselines_rieth.json --out docs/results/plant_bytes_baselines_rieth.md
```
