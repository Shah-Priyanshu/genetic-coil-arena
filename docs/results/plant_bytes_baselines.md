# Plant floor: diagnosis quality vs uplink bytes (R8, step 1)

- Date: 2026-10-23; on top of base commit 1fc4c37; Python 3.13.7; Intel64 Family 6 Model 183 Stepping 1, GenuineIntel; 4 worker processes, 2.2 min wall clock
- Data: Tennessee Eastman Process, Braatz-group benchmark d00-d21 (https://web.mit.edu/braatzgroup/TE_process.zip, retrieved 2026-10-23). Licence: University of Illinois / Large Scale Systems Research Laboratory (Prof. Richard Braatz), 1998-2002: free of charge, permission to deal with the Software and data without restriction (copy, modify, merge, publish, distribute, sublicense, sell), subject to keeping the copyright notice and disclaimers (NCSA-style open-source licence, see readme.txt in the archive). Files are not committed; `python scripts/fetch_tep.py` downloads them to a cache outside the repo and checks every file's SHA-256 and shape.
- Seeds: 10 (0..9); every number is mean ± 95% t-CI over seeds (`evaluation.stats.mean_ci`). A seed draws a different 80% subsample of the training blocks and a different classifier initialisation / noise stream. **The CI does not cover run-to-run plant variability**: the held-out test is one recorded run per fault (the same for every seed), see Caveats.
- Protocol: window 20 samples (60 min), one uplink and one diagnosis every 10 samples (30 min); test runs are 960 samples with the fault from sample 160 (95 windows per run); the centre never sees the clock or the onset. Window label = the run's fault once the window contains a faulty sample (>= 1 faulty sample), else normal.
- Centre: softmax regression (L2 0.01, class-balanced) on the decoded messages, trained on the training split at stride 4; normal margin calibrated by time-blocked cross-validation on training data to 5% false alarms on held-out normal windows (same target for every encoding).
- LoRaWAN EU868 reference: 51 B at SF12 is 2.79 s on air, so a 1% duty cycle allows one uplink per 279 s per node; our uplinks are 1800 s apart, so every budget here is legal at SF12. JSON of the same readings would not fit (a 4-variable JSON object is already tens of bytes).

## Variable mapping (52 TEP variables to 4 nodes)

Each node owns the sensors and actuators of one unit operation or service; analysers are attached to the unit that the sampled stream leaves.

- **reactor** (18 variables): XMEAS(1) A feed (stream 1) [kscmh]; XMEAS(2) D feed (stream 2) [kg/h]; XMEAS(3) E feed (stream 3) [kg/h]; XMEAS(4) A and C feed (stream 4) [kscmh]; XMEAS(6) Reactor feed rate (stream 6) [kscmh]; XMEAS(7) Reactor pressure [kPa gauge]; XMEAS(8) Reactor level [%]; XMEAS(9) Reactor temperature [deg C]; XMEAS(23) Reactor feed analysis, component A (stream 6) [mol%]; XMEAS(24) Reactor feed analysis, component B (stream 6) [mol%]; XMEAS(25) Reactor feed analysis, component C (stream 6) [mol%]; XMEAS(26) Reactor feed analysis, component D (stream 6) [mol%]; XMEAS(27) Reactor feed analysis, component E (stream 6) [mol%]; XMEAS(28) Reactor feed analysis, component F (stream 6) [mol%]; XMV(1) D feed flow (stream 2) [%]; XMV(2) E feed flow (stream 3) [%]; XMV(3) A feed flow (stream 1) [%]; XMV(4) A and C feed flow (stream 4) [%]
- **separator** (15 variables): XMEAS(10) Purge rate (stream 9) [kscmh]; XMEAS(11) Product separator temperature [deg C]; XMEAS(12) Product separator level [%]; XMEAS(13) Product separator pressure [kPa gauge]; XMEAS(14) Product separator underflow (stream 10) [m3/h]; XMEAS(29) Purge gas analysis, component A (stream 9) [mol%]; XMEAS(30) Purge gas analysis, component B (stream 9) [mol%]; XMEAS(31) Purge gas analysis, component C (stream 9) [mol%]; XMEAS(32) Purge gas analysis, component D (stream 9) [mol%]; XMEAS(33) Purge gas analysis, component E (stream 9) [mol%]; XMEAS(34) Purge gas analysis, component F (stream 9) [mol%]; XMEAS(35) Purge gas analysis, component G (stream 9) [mol%]; XMEAS(36) Purge gas analysis, component H (stream 9) [mol%]; XMV(6) Purge valve (stream 9) [%]; XMV(7) Separator pot liquid flow (stream 10) [%]
- **stripper** (10 variables): XMEAS(15) Stripper level [%]; XMEAS(16) Stripper pressure [kPa gauge]; XMEAS(17) Stripper underflow (stream 11) [m3/h]; XMEAS(18) Stripper temperature [deg C]; XMEAS(37) Product analysis, component D (stream 11) [mol%]; XMEAS(38) Product analysis, component E (stream 11) [mol%]; XMEAS(39) Product analysis, component F (stream 11) [mol%]; XMEAS(40) Product analysis, component G (stream 11) [mol%]; XMEAS(41) Product analysis, component H (stream 11) [mol%]; XMV(8) Stripper liquid product flow (stream 11) [%]
- **compressor_utilities** (9 variables): XMEAS(5) Recycle flow (stream 8) [kscmh]; XMEAS(19) Stripper steam flow [kg/h]; XMEAS(20) Compressor work [kW]; XMEAS(21) Reactor cooling water outlet temperature [deg C]; XMEAS(22) Separator cooling water outlet temperature [deg C]; XMV(5) Compressor recycle valve [%]; XMV(9) Stripper steam valve [%]; XMV(10) Reactor cooling water flow [%]; XMV(11) Condenser cooling water flow [%]

## Summary (18 faults)

- Best hand design by macro-F1 at own calibrated threshold (8 B: features 0.72, 16 B: features 0.76, 51 B: features 0.72); oracle 0.69; silent 0.00; random bytes 0.00.
- Best hand design by macro-F1 at matched 5% test false alarms (8 B: features 0.71, 16 B: features 0.76, 51 B: features 0.72); oracle 0.70; silent 0.00; random bytes 0.01.
- Macro-F1 of (c) at matched false alarms by budget: 1 B 0.47, 8 B 0.71, 16 B 0.76, 51 B 0.72; the fixed one-byte alarm code (a): 0.38.
- The oracle (0.70) is *below* the best byte-limited design (0.76) at matched false alarms, so it is not an upper bound for this centre: a linear classifier fitted on a few hundred strongly autocorrelated windows per class does worse with all the fields than with a compact subset (a likely cause, not tested here). Bandwidth does not limit this centre beyond about 16 B; a stronger centre or more training data (Rieth et al.) may change that.

## Summary (21 faults)

- Best hand design by macro-F1 at own calibrated threshold (8 B: features 0.62, 16 B: features 0.64, 51 B: features 0.60); oracle 0.58; silent 0.00; random bytes 0.00.
- Best hand design by macro-F1 at matched 5% test false alarms (8 B: features 0.60, 16 B: features 0.62, 51 B: features 0.58); oracle 0.56; silent 0.00; random bytes 0.00.
- Macro-F1 of (c) at matched false alarms by budget: 1 B 0.40, 8 B 0.60, 16 B 0.62, 51 B 0.58; the fixed one-byte alarm code (a): 0.30.
- The oracle (0.56) is *below* the best byte-limited design (0.62) at matched false alarms, so it is not an upper bound for this centre: a linear classifier fitted on a few hundred strongly autocorrelated windows per class does worse with all the fields than with a compact subset (a likely cause, not tested here). Bandwidth does not limit this centre beyond about 16 B; a stronger centre or more training data (Rieth et al.) may change that.

## Results: 18 diagnosable faults (faults 3, 9 and 15 left out: no method separates them from normal operation in this data)

### Macro-F1 over the fault classes (higher is better)

| Encoding | 0 B | 1 B | 8 B | 16 B | 51 B |
|---|---|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | - | 0.479 ± 0.008 | 0.479 ± 0.008 | 0.479 ± 0.008 | 0.479 ± 0.008 |
| (b) quantised summary stats (mean, std, max, min) | - | 0.202 ± 0.019 | 0.574 ± 0.012 | 0.595 ± 0.019 | 0.603 ± 0.027 |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | - | 0.487 ± 0.018 | 0.719 ± 0.006 | 0.759 ± 0.006 | 0.725 ± 0.008 |
| (d) silent (0 B, always reports normal) | 0.000 ± 0.000 | - | - | - | - |
| (e) oracle (all raw variables, no byte limit) | 0.692 ± 0.016 | - | - | - | - |
| control: random bytes | - | 0.002 ± 0.001 | 0.002 ± 0.001 | 0.001 ± 0.001 | 0.001 ± 0.001 |

Mean ± 95% CI over 10 seeds. F1 of each of the 18 fault classes from the confusion of all test windows (normal windows mistaken for a fault count against that class), then averaged. Random-guess level is about 0.05.

### Detection delay in samples (lower is better; 1 sample = 3 min)

| Encoding | 0 B | 1 B | 8 B | 16 B | 51 B |
|---|---|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | - | 18.4 ± 0.6 | 18.4 ± 0.6 | 18.4 ± 0.6 | 18.4 ± 0.6 |
| (b) quantised summary stats (mean, std, max, min) | - | 196.1 ± 16.8 | 52.6 ± 2.8 | 55.3 ± 12.1 | 44.9 ± 7.8 |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | - | 13.4 ± 0.8 | 20.2 ± 3.4 | 24.4 ± 2.1 | 26.1 ± 2.5 |
| (d) silent (0 B, always reports normal) | never | - | - | - | - |
| (e) oracle (all raw variables, no byte limit) | 31.8 ± 2.1 | - | - | - | - |
| control: random bytes | - | 295.3 ± 65.6 | 330.2 ± 77.4 | 320.5 ± 55.6 | 405.4 ± 84.9 |

Samples from the fault onset to the end of the first window in which the centre raises any alarm, averaged over the fault runs that were detected (the minimum is 10, one stride). `never` = no run detected.

### Detection rate (higher is better)

| Encoding | 0 B | 1 B | 8 B | 16 B | 51 B |
|---|---|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | - | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| (b) quantised summary stats (mean, std, max, min) | - | 0.62 ± 0.04 | 0.99 ± 0.01 | 0.98 ± 0.02 | 0.99 ± 0.01 |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | - | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| (d) silent (0 B, always reports normal) | 0.00 ± 0.00 | - | - | - | - |
| (e) oracle (all raw variables, no byte limit) | 1.00 ± 0.00 | - | - | - | - |
| control: random bytes | - | 0.57 ± 0.26 | 0.61 ± 0.19 | 0.52 ± 0.18 | 0.34 ± 0.19 |

Share of the fault runs in which the centre ever alarmed after the onset.

### False-alarm rate on normal windows (lower is better)

| Encoding | 0 B | 1 B | 8 B | 16 B | 51 B |
|---|---|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | - | 0.211 ± 0.019 | 0.211 ± 0.019 | 0.211 ± 0.019 | 0.211 ± 0.019 |
| (b) quantised summary stats (mean, std, max, min) | - | 0.005 ± 0.003 | 0.017 ± 0.008 | 0.011 ± 0.007 | 0.015 ± 0.009 |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | - | 0.625 ± 0.068 | 0.241 ± 0.056 | 0.105 ± 0.045 | 0.100 ± 0.044 |
| (d) silent (0 B, always reports normal) | 0.000 ± 0.000 | - | - | - | - |
| (e) oracle (all raw variables, no byte limit) | 0.061 ± 0.031 | - | - | - | - |
| control: random bytes | - | 0.016 ± 0.011 | 0.020 ± 0.016 | 0.008 ± 0.004 | 0.007 ± 0.006 |

Share of normal test windows (the normal run plus the pre-onset part of every fault run) that raise an alarm. The calibration target was 5% on held-out training windows; the test value is higher because the test runs sit at different offsets from the single 500-sample training run.

### Same decisions at a matched 5% false-alarm rate on the test normal windows

Each centre's alarm threshold is moved so that exactly 5% of the test normal windows alarm (the fault class it reports is unchanged). This looks at the test normal labels, so it is not a deployable setting; it puts every encoding at one operating point so that macro-F1 and delay compare information content rather than how well the training calibration transferred. The silent row is unchanged (it never alarms).

**Macro-F1 at matched false alarms (higher is better)**

| Encoding | 0 B | 1 B | 8 B | 16 B | 51 B |
|---|---|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | - | 0.376 ± 0.014 | 0.376 ± 0.014 | 0.376 ± 0.014 | 0.376 ± 0.014 |
| (b) quantised summary stats (mean, std, max, min) | - | 0.209 ± 0.020 | 0.597 ± 0.003 | 0.659 ± 0.010 | 0.657 ± 0.009 |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | - | 0.468 ± 0.018 | 0.707 ± 0.008 | 0.759 ± 0.004 | 0.721 ± 0.003 |
| (d) silent (0 B, always reports normal) | 0.000 ± 0.000 | - | - | - | - |
| (e) oracle (all raw variables, no byte limit) | 0.699 ± 0.009 | - | - | - | - |
| control: random bytes | - | 0.005 ± 0.002 | 0.007 ± 0.003 | 0.006 ± 0.003 | 0.006 ± 0.002 |

Mean ± 95% CI over 10 seeds.

**Detection delay in samples at matched false alarms**

| Encoding | 0 B | 1 B | 8 B | 16 B | 51 B |
|---|---|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | - | 98.4 ± 13.5 | 98.4 ± 13.5 | 98.4 ± 13.5 | 98.4 ± 13.5 |
| (b) quantised summary stats (mean, std, max, min) | - | 154.7 ± 18.7 | 46.8 ± 2.7 | 28.2 ± 1.8 | 32.1 ± 1.9 |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | - | 67.3 ± 12.9 | 35.2 ± 3.1 | 26.4 ± 0.9 | 27.3 ± 1.9 |
| (d) silent (0 B, always reports normal) | never | - | - | - | - |
| (e) oracle (all raw variables, no byte limit) | 31.3 ± 1.2 | - | - | - | - |
| control: random bytes | - | 176.1 ± 30.3 | 194.5 ± 43.2 | 164.2 ± 35.2 | 156.2 ± 24.8 |

Mean over the detected fault runs; see the detection rate below.

**Detection rate at matched false alarms**

| Encoding | 0 B | 1 B | 8 B | 16 B | 51 B |
|---|---|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | - | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| (b) quantised summary stats (mean, std, max, min) | - | 0.99 ± 0.01 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | - | 0.99 ± 0.02 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| (d) silent (0 B, always reports normal) | 0.00 ± 0.00 | - | - | - | - |
| (e) oracle (all raw variables, no byte limit) | 1.00 ± 0.00 | - | - | - | - |
| control: random bytes | - | 0.99 ± 0.02 | 0.99 ± 0.02 | 0.99 ± 0.01 | 0.97 ± 0.03 |

Share of fault runs detected.

### Mean bytes actually used per node message

| Encoding | 0 B | 1 B | 8 B | 16 B | 51 B |
|---|---|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | - | 1.0 | 1.0 | 1.0 | 1.0 |
| (b) quantised summary stats (mean, std, max, min) | - | 1.0 | 8.0 | 16.0 | 35.5 |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | - | 1.0 | 8.0 | 16.0 | 31.7 |
| (d) silent (0 B, always reports normal) | 0.0 | - | - | - | - |
| (e) oracle (all raw variables, no byte limit) | 0.0 | - | - | - | - |
| control: random bytes | - | 1.0 | 8.0 | 16.0 | 51.0 |

## Results: all 21 faults

### Macro-F1 over the fault classes (higher is better)

| Encoding | 0 B | 1 B | 8 B | 16 B | 51 B |
|---|---|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | - | 0.400 ± 0.005 | 0.400 ± 0.005 | 0.400 ± 0.005 | 0.400 ± 0.005 |
| (b) quantised summary stats (mean, std, max, min) | - | 0.172 ± 0.023 | 0.482 ± 0.018 | 0.486 ± 0.017 | 0.490 ± 0.020 |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | - | 0.417 ± 0.013 | 0.620 ± 0.005 | 0.641 ± 0.012 | 0.604 ± 0.012 |
| (d) silent (0 B, always reports normal) | 0.000 ± 0.000 | - | - | - | - |
| (e) oracle (all raw variables, no byte limit) | 0.580 ± 0.018 | - | - | - | - |
| control: random bytes | - | 0.001 ± 0.001 | 0.002 ± 0.001 | 0.000 ± 0.001 | 0.000 ± 0.001 |

Mean ± 95% CI over 10 seeds. F1 of each of the 21 fault classes from the confusion of all test windows (normal windows mistaken for a fault count against that class), then averaged. Random-guess level is about 0.05.

### Detection delay in samples (lower is better; 1 sample = 3 min)

| Encoding | 0 B | 1 B | 8 B | 16 B | 51 B |
|---|---|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | - | 19.2 ± 0.5 | 19.2 ± 0.5 | 19.2 ± 0.5 | 19.2 ± 0.5 |
| (b) quantised summary stats (mean, std, max, min) | - | 190.9 ± 20.2 | 65.3 ± 12.8 | 80.5 ± 11.8 | 78.8 ± 14.6 |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | - | 14.5 ± 1.9 | 25.9 ± 4.0 | 32.6 ± 3.9 | 37.2 ± 7.3 |
| (d) silent (0 B, always reports normal) | never | - | - | - | - |
| (e) oracle (all raw variables, no byte limit) | 46.0 ± 15.1 | - | - | - | - |
| control: random bytes | - | 312.3 ± 58.0 | 291.5 ± 77.8 | 314.4 ± 46.4 | 388.5 ± 77.5 |

Samples from the fault onset to the end of the first window in which the centre raises any alarm, averaged over the fault runs that were detected (the minimum is 10, one stride). `never` = no run detected.

### Detection rate (higher is better)

| Encoding | 0 B | 1 B | 8 B | 16 B | 51 B |
|---|---|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | - | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| (b) quantised summary stats (mean, std, max, min) | - | 0.60 ± 0.06 | 0.91 ± 0.05 | 0.90 ± 0.04 | 0.98 ± 0.02 |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | - | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| (d) silent (0 B, always reports normal) | 0.00 ± 0.00 | - | - | - | - |
| (e) oracle (all raw variables, no byte limit) | 1.00 ± 0.00 | - | - | - | - |
| control: random bytes | - | 0.58 ± 0.24 | 0.66 ± 0.20 | 0.43 ± 0.16 | 0.33 ± 0.16 |

Share of the fault runs in which the centre ever alarmed after the onset.

### False-alarm rate on normal windows (lower is better)

| Encoding | 0 B | 1 B | 8 B | 16 B | 51 B |
|---|---|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | - | 0.246 ± 0.019 | 0.246 ± 0.019 | 0.246 ± 0.019 | 0.246 ± 0.019 |
| (b) quantised summary stats (mean, std, max, min) | - | 0.003 ± 0.003 | 0.020 ± 0.015 | 0.015 ± 0.006 | 0.026 ± 0.015 |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | - | 0.627 ± 0.073 | 0.243 ± 0.079 | 0.103 ± 0.037 | 0.091 ± 0.020 |
| (d) silent (0 B, always reports normal) | 0.000 ± 0.000 | - | - | - | - |
| (e) oracle (all raw variables, no byte limit) | 0.092 ± 0.035 | - | - | - | - |
| control: random bytes | - | 0.017 ± 0.011 | 0.020 ± 0.013 | 0.009 ± 0.006 | 0.008 ± 0.008 |

Share of normal test windows (the normal run plus the pre-onset part of every fault run) that raise an alarm. The calibration target was 5% on held-out training windows; the test value is higher because the test runs sit at different offsets from the single 500-sample training run.

### Same decisions at a matched 5% false-alarm rate on the test normal windows

Each centre's alarm threshold is moved so that exactly 5% of the test normal windows alarm (the fault class it reports is unchanged). This looks at the test normal labels, so it is not a deployable setting; it puts every encoding at one operating point so that macro-F1 and delay compare information content rather than how well the training calibration transferred. The silent row is unchanged (it never alarms).

**Macro-F1 at matched false alarms (higher is better)**

| Encoding | 0 B | 1 B | 8 B | 16 B | 51 B |
|---|---|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | - | 0.303 ± 0.009 | 0.303 ± 0.009 | 0.303 ± 0.009 | 0.303 ± 0.009 |
| (b) quantised summary stats (mean, std, max, min) | - | 0.179 ± 0.021 | 0.507 ± 0.006 | 0.538 ± 0.008 | 0.527 ± 0.008 |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | - | 0.398 ± 0.013 | 0.602 ± 0.005 | 0.622 ± 0.009 | 0.583 ± 0.007 |
| (d) silent (0 B, always reports normal) | 0.000 ± 0.000 | - | - | - | - |
| (e) oracle (all raw variables, no byte limit) | 0.556 ± 0.011 | - | - | - | - |
| control: random bytes | - | 0.003 ± 0.001 | 0.004 ± 0.002 | 0.005 ± 0.001 | 0.005 ± 0.002 |

Mean ± 95% CI over 10 seeds.

**Detection delay in samples at matched false alarms**

| Encoding | 0 B | 1 B | 8 B | 16 B | 51 B |
|---|---|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | - | 129.5 ± 13.2 | 129.5 ± 13.2 | 129.5 ± 13.2 | 129.5 ± 13.2 |
| (b) quantised summary stats (mean, std, max, min) | - | 125.3 ± 20.1 | 74.8 ± 10.7 | 51.8 ± 8.6 | 57.7 ± 5.8 |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | - | 78.3 ± 16.8 | 47.8 ± 9.2 | 44.6 ± 7.6 | 46.5 ± 6.3 |
| (d) silent (0 B, always reports normal) | never | - | - | - | - |
| (e) oracle (all raw variables, no byte limit) | 60.1 ± 11.1 | - | - | - | - |
| control: random bytes | - | 194.2 ± 39.6 | 173.3 ± 27.0 | 184.9 ± 35.6 | 181.4 ± 43.9 |

Mean over the detected fault runs; see the detection rate below.

**Detection rate at matched false alarms**

| Encoding | 0 B | 1 B | 8 B | 16 B | 51 B |
|---|---|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | - | 0.99 ± 0.02 | 0.99 ± 0.02 | 0.99 ± 0.02 | 0.99 ± 0.02 |
| (b) quantised summary stats (mean, std, max, min) | - | 1.00 ± 0.01 | 0.99 ± 0.01 | 1.00 ± 0.01 | 1.00 ± 0.00 |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | - | 1.00 ± 0.01 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| (d) silent (0 B, always reports normal) | 0.00 ± 0.00 | - | - | - | - |
| (e) oracle (all raw variables, no byte limit) | 1.00 ± 0.00 | - | - | - | - |
| control: random bytes | - | 0.99 ± 0.02 | 0.98 ± 0.02 | 0.99 ± 0.02 | 0.98 ± 0.04 |

Share of fault runs detected.

### Mean bytes actually used per node message

| Encoding | 0 B | 1 B | 8 B | 16 B | 51 B |
|---|---|---|---|---|---|
| (a) threshold alarms (1 B alarm byte per node) | - | 1.0 | 1.0 | 1.0 | 1.0 |
| (b) quantised summary stats (mean, std, max, min) | - | 1.0 | 8.0 | 16.0 | 35.5 |
| (c) quantised hand features (PCA SPE/T2/scores, FFT bands) | - | 1.0 | 8.0 | 16.0 | 31.7 |
| (d) silent (0 B, always reports normal) | 0.0 | - | - | - | - |
| (e) oracle (all raw variables, no byte limit) | 0.0 | - | - | - | - |
| control: random bytes | - | 1.0 | 8.0 | 16.0 | 51.0 |

## CommArena harness cross-check (16 B, 18 faults)

The same cells through `evaluation.run_experiment`: each system against the mandatory baselines (silent centre, random-bytes control) with Mann-Whitney tests (Holm-corrected), then the message ablations on the nodes' uplink (`zero`, `shuffle_batch`, `swap_receivers`, `random_matched`; paired Wilcoxon, `listening` = the performance drop's CI lies above zero). Score = macro-F1. Mean macro-F1 here equals the pool run's for every system: yes.

```
Task: plant_diagnosis_18faults_16B  (seeds: 10)
Baselines: silent, random_bytes

condition     role      n   mean ± 95% CI  IQM [95% CI]        
------------  --------  --  -------------  --------------------
alarms        system    10  0.479 ± 0.008  0.479 [0.472, 0.486]
summary       system    10  0.595 ± 0.019  0.593 [0.574, 0.616]
features      system    10  0.759 ± 0.006  0.761 [0.754, 0.765]
silent        baseline  10  0.000 ± 0.000  0.000 [0.000, 0.000]
random_bytes  baseline  10  0.001 ± 0.001  0.000 [0.000, 0.001]

system    intervention    ablated mean  drop [95% CI]         p (Holm)  A12    listening
--------  --------------  ------------  --------------------  --------  -----  ---------
alarms    zero            0.000         0.479 [0.471, 0.487]  0.008     1.000  yes      
alarms    shuffle_batch   0.034         0.445 [0.434, 0.456]  0.008     1.000  yes      
alarms    swap_receivers  0.035         0.444 [0.435, 0.453]  0.008     1.000  yes      
alarms    random_matched  0.045         0.434 [0.424, 0.443]  0.008     1.000  yes      
summary   zero            0.005         0.590 [0.572, 0.609]  0.008     1.000  yes      
summary   shuffle_batch   0.018         0.577 [0.558, 0.596]  0.008     1.000  yes      
summary   swap_receivers  0.045         0.550 [0.528, 0.573]  0.008     1.000  yes      
summary   random_matched  0.034         0.561 [0.540, 0.582]  0.008     1.000  yes      
features  zero            0.000         0.759 [0.754, 0.765]  0.008     1.000  yes      
features  shuffle_batch   0.024         0.735 [0.729, 0.741]  0.008     1.000  yes      
features  swap_receivers  0.048         0.711 [0.704, 0.719]  0.008     1.000  yes      
features  random_matched  0.042         0.717 [0.712, 0.723]  0.008     1.000  yes      

comparison                mean diff  p      p (Holm)  A12  
------------------------  ---------  -----  --------  -----
alarms vs silent          0.479      0.000  0.000     1.000
alarms vs random_bytes    0.478      0.000  0.000     1.000
summary vs silent         0.595      0.000  0.000     1.000
summary vs random_bytes   0.594      0.000  0.000     1.000
features vs silent        0.759      0.000  0.000     1.000
features vs random_bytes  0.759      0.000  0.000     1.000
```

## Caveats

- One recorded test run per fault (and one 500-sample normal training run). The seeds vary training only, so the CIs understate the uncertainty about a new plant run; the 500-run Rieth et al. 2017 data set (doi:10.7910/DVN/6C3JR1) would give run-level CIs.
- Fault classes 3, 9 and 15 are indistinguishable from normal operation in this data; the all-21-fault tables show what they cost.
- Slow variables carry large run-to-run offsets. The pre-onset means of stripper temperature, stripper steam flow and stripper steam valve range from -0.5 to +1.5 training standard deviations across the 21 fault test runs (about +0.4 to +0.5 in the normal test run), so a threshold calibrated on the single 500-sample training run does not transfer cleanly: realised false-alarm rates exceed the calibration target and differ a lot between encodings (macro-F1 and delay at each centre's own threshold trade against that rate; hence the matched-false-alarm tables).
- The alarm design is a fixed 1 B code, so its row does not change with the budget; the 0 B column of an encoding means no message, the same as (d).
- The oracle uses the same centre family (softmax regression) on unquantised statistics of every variable plus every PCA/band feature (c) can send. It holds all the information but is a bound on nothing: with so few independent training windows a wider input tends to over-fit, so it can score below the compact feature set (c) (see the Summary for whether it does). Compare encodings at the matched false-alarm rate, where the thresholds agree.
- The classifier L2 strength was fixed at one value after a coarse look at {1e-3, 1e-2, 1e-1} that moved macro-F1 by under 0.03; it is not tuned per encoding.

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
- IDV(21): Valve for stream 4 fixed at its steady-state position, constant position

## Reproduce

```bash
python scripts/fetch_tep.py
python scripts/plant_baselines.py --seeds 10 --workers 4 --out docs/results/plant_bytes_baselines.md
```
