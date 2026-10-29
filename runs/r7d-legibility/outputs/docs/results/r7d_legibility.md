# R7d: can a monitor read and write the squad's learned language? (goal G12)

**G12 verdict: PASS.** On held-out goals, per-squad decoders fitted on training-goal episodes recover the exact eater set in 94.1% ± 1.5 of episodes (pass ≥ 90%, kill < 60%), and the decoded writer steers a snake with efficacy 0.74 ± 0.07 (pass ≥ 0.50, kill < 0.25); n = 34 (b) runs with best training-goal validation ≥ 12.

Decoders read the first 30 steps of each live receiver's symbols; fit: 40 episodes per training goal; test: 50 per held-out goal; 10 random writer pairs per run for the null. Means ± 95% t-CI over training runs.

## Read and write, held-out goals

| Runs | n | Eater set exact | Shuffled-label null | Single symbol (bal. acc.) | Caution | Whole spec | Write efficacy | Random writers |
|---|---|---|---|---|---|---|---|---|
| (b), G12 sample (n = 34) | 34 | 94.1 ± 1.5% | 52.2 ± 14.1% | 79.4 ± 2.5% | 28.9 ± 4.1% | 26.6 ± 4.1% | 0.74 ± 0.07 | 0.00 ± 0.02 |
| (b), all runs | 36 | 94.4 ± 1.5% | 49.3 ± 13.9% | 80.4 ± 2.8% | 28.2 ± 4.0% | 26.1 ± 3.9% | 0.75 ± 0.07 | 0.01 ± 0.02 |
| (c) goal-centre+goal-spec | 16 | 60.4 ± 13.3% | 29.6 ± 16.5% | 63.3 ± 5.6% | 36.6 ± 7.4% | 20.9 ± 6.6% | 0.35 ± 0.08 | -0.01 ± 0.02 |
| (d) hand-goal-centre+learned-spec | 16 | 100.0 ± 0.0% | 67.1 ± 24.9% | 100.0 ± 0.0% | 17.0 ± 0.2% | 17.0 ± 0.2% | 0.98 ± 0.01 | -0.00 ± 0.02 |
| (a) blind-centre+learned-spec | 16 | 16.3 ± 0.9% | 16.8 ± 1.4% | 50.0 ± 0.1% | 32.5 ± 1.0% | 5.0 ± 1.4% | nan ± nan | nan ± nan |

Food of the steered snake (mean per episode) against the intact squad:

| Runs | Written "eat" | Written "do not eat" | Intact eaters | Intact non-eaters |
|---|---|---|---|---|
| (b), G12 sample (n = 34) | 8.70 ± 0.80 | 0.20 ± 0.04 | 11.73 ± 0.27 | 0.24 ± 0.05 |
| (b), all runs | 8.49 ± 0.81 | 0.19 ± 0.04 | 11.36 ± 0.58 | 0.23 ± 0.05 |
| (c) goal-centre+goal-spec | 5.79 ± 1.02 | 1.71 ± 0.92 | 11.65 ± 0.45 | 0.16 ± 0.03 |
| (d) hand-goal-centre+learned-spec | 12.15 ± 0.64 | 0.18 ± 0.04 | 12.43 ± 0.66 | 0.20 ± 0.04 |
| (a) blind-centre+learned-spec | 0.45 ± 0.16 | 0.46 ± 0.15 | 0.44 ± 0.12 | 0.45 ± 0.14 |

## Dialects

Among the 34 sample squads there are 33 distinct writer tables. A squad's own decoder recovers 94.1% of its held-out eater sets; another squad's decoder 23.8% on average (best pair 100.0%).

## Runs

| Condition | Run | Best validation | Eater set exact | Write efficacy | Writer, eat (situations 0-7) | Writer, do not eat |
|---|---|---|---|---|---|---|
| (a) blind-centre+learned-spec | 0 | -3.5 | 15.3% | nan | 44444440 | 44444440 |
| (a) blind-centre+learned-spec | 1 | -3.4 | 18.7% | -14.89 | 37775555 | 77775555 |
| (a) blind-centre+learned-spec | 2 | -3.5 | 14.7% | -4.22 | 44444544 | 44444444 |
| (a) blind-centre+learned-spec | 3 | -3.5 | 16.0% | 0.00 | 77777777 | 77777777 |
| (a) blind-centre+learned-spec | 4 | -3.3 | 17.0% | nan | 11111111 | 11111111 |
| (a) blind-centre+learned-spec | 5 | -3.1 | 16.0% | nan | 22222222 | 22222222 |
| (a) blind-centre+learned-spec | 6 | -3.2 | 17.0% | nan | 22252121 | 22252121 |
| (a) blind-centre+learned-spec | 7 | -3.6 | 17.3% | 0.00 | 66323366 | 66323366 |
| (a) blind-centre+learned-spec | 8 | -3.5 | 15.0% | nan | 66666666 | 66666666 |
| (a) blind-centre+learned-spec | 9 | -3.6 | 12.3% | 0.00 | 44444444 | 44444444 |
| (a) blind-centre+learned-spec | 10 | -3.7 | 16.7% | 0.00 | 55555555 | 55555555 |
| (a) blind-centre+learned-spec | 11 | -3.6 | 16.7% | 0.00 | 66666666 | 66666666 |
| (a) blind-centre+learned-spec | 12 | -3.2 | 19.3% | nan | 14111111 | 14111111 |
| (a) blind-centre+learned-spec | 13 | -3.5 | 17.3% | nan | 66444266 | 66444466 |
| (a) blind-centre+learned-spec | 14 | -3.7 | 16.3% | 0.00 | 11111117 | 11111117 |
| (a) blind-centre+learned-spec | 15 | -3.5 | 15.0% | nan | 55555553 | 55555553 |
| (b) goal-centre+learned-spec | 0 | 22.4 | 91.0% | 0.85 | 40111144 | 00000000 |
| (b) goal-centre+learned-spec | 1 | 18.6 | 99.7% | 0.94 | 11777711 | 55555555 |
| (b) goal-centre+learned-spec | 2 | 20.4 | 90.7% | 0.92 | 66222266 | 22222222 |
| (b) goal-centre+learned-spec | 3 | 20.2 | 85.0% | 0.51 | 55111115 | 11111111 |
| (b) goal-centre+learned-spec | 4 | 21.8 | 84.3% | 0.71 | 66565555 | 44444444 |
| (b) goal-centre+learned-spec | 5 | 21.8 | 87.0% | 0.91 | 55000055 | 00000000 |
| (b) goal-centre+learned-spec | 6 | 20.0 | 87.0% | 0.50 | 66333333 | 66666666 |
| (b) goal-centre+learned-spec | 7 | 21.7 | 95.0% | 0.67 | 66060000 | 00000000 |
| (b) goal-centre+learned-spec | 8 | 19.9 | 94.0% | 0.57 | 77333337 | 77777777 |
| (b) goal-centre+learned-spec | 9 | 21.6 | 100.0% | 0.70 | 33331111 | 55555155 |
| (b) goal-centre+learned-spec | 10 | 5.7 | 99.7% | 1.00 | 44444444 | 33333333 |
| (b) goal-centre+learned-spec | 11 | 20.9 | 97.7% | 0.63 | 00037777 | 77777777 |
| (b) goal-centre+learned-spec | 12 | 18.1 | 97.0% | 0.95 | 33335555 | 55555555 |
| (b) goal-centre+learned-spec | 13 | 21.5 | 94.3% | 0.95 | 55551111 | 11111111 |
| (b) goal-centre+learned-spec | 14 | 20.3 | 94.0% | 0.75 | 11110000 | 00000000 |
| (b) goal-centre+learned-spec | 15 | 8.2 | 99.7% | 0.98 | 11111111 | 22222222 |
| (b) goal-centre+learned-spec | 16 | 21.7 | 94.7% | 0.62 | 66363333 | 33333333 |
| (b) goal-centre+learned-spec | 17 | 17.8 | 96.0% | 0.92 | 66665555 | 55555555 |
| (b) goal-centre+learned-spec | 18 | 22.2 | 100.0% | 0.91 | 44550000 | 22252222 |
| (b) goal-centre+learned-spec | 19 | 20.2 | 90.0% | 0.47 | 11313333 | 11111111 |
| (b) goal-centre+learned-spec | 20 | 22.5 | 92.0% | 0.70 | 66464444 | 44444444 |
| (b) goal-centre+learned-spec | 21 | 21.4 | 92.7% | 0.46 | 33555533 | 55555555 |
| (b) goal-centre+learned-spec | 22 | 18.3 | 96.3% | 0.63 | 00606666 | 00000000 |
| (b) goal-centre+learned-spec | 23 | 21.5 | 91.0% | 1.14 | 22225555 | 33333333 |
| (b) goal-centre+learned-spec | 24 | 20.3 | 95.7% | 0.60 | 66161111 | 11111111 |
| (b) goal-centre+learned-spec | 25 | 20.5 | 95.3% | 0.73 | 66662222 | 22222222 |
| (b) goal-centre+learned-spec | 26 | 20.7 | 95.7% | 0.15 | 11114411 | 44444444 |
| (b) goal-centre+learned-spec | 27 | 20.4 | 100.0% | 0.84 | 66667777 | 36333333 |
| (b) goal-centre+learned-spec | 28 | 21.2 | 95.0% | 0.97 | 66777766 | 77777777 |
| (b) goal-centre+learned-spec | 29 | 19.3 | 99.0% | 0.83 | 00005555 | 55555555 |
| (b) goal-centre+learned-spec | 30 | 20.2 | 90.3% | 0.67 | 00006666 | 66666666 |
| (b) goal-centre+learned-spec | 31 | 19.2 | 97.7% | 0.94 | 44555544 | 55555555 |
| (b) goal-centre+learned-spec | 32 | 18.9 | 95.0% | 0.59 | 11115555 | 11111111 |
| (b) goal-centre+learned-spec | 33 | 22.3 | 92.0% | 0.99 | 44555544 | 55555555 |
| (b) goal-centre+learned-spec | 34 | 22.5 | 99.3% | 0.76 | 55557777 | 77777777 |
| (b) goal-centre+learned-spec | 35 | 19.1 | 95.7% | 0.66 | 66767777 | 77777777 |
| (c) goal-centre+goal-spec | 0 | 21.7 | 81.3% | 0.71 | 55222255 | 22222222 |
| (c) goal-centre+goal-spec | 1 | 20.1 | 38.7% | 0.34 | 00707777 | 00000700 |
| (c) goal-centre+goal-spec | 2 | 21.7 | 88.0% | 0.54 | 33335555 | 55555555 |
| (c) goal-centre+goal-spec | 3 | 24.2 | 49.0% | 0.45 | 66444466 | 66666466 |
| (c) goal-centre+goal-spec | 4 | 20.2 | 45.7% | 0.33 | 11313333 | 11113311 |
| (c) goal-centre+goal-spec | 5 | 19.6 | 88.7% | 0.33 | 33030000 | 03000000 |
| (c) goal-centre+goal-spec | 6 | 19.8 | 60.7% | 0.01 | 11115555 | 11515555 |
| (c) goal-centre+goal-spec | 7 | 22.1 | 25.7% | 0.44 | 66111166 | 66666166 |
| (c) goal-centre+goal-spec | 8 | 19.2 | 86.3% | 0.25 | 33355533 | 53555555 |
| (c) goal-centre+goal-spec | 9 | 21.1 | 38.3% | 0.27 | 00333330 | 00000300 |
| (c) goal-centre+goal-spec | 10 | 19.9 | 43.3% | 0.29 | 22020000 | 22222022 |
| (c) goal-centre+goal-spec | 11 | 18.5 | 71.7% | 0.34 | 22020000 | 22222222 |
| (c) goal-centre+goal-spec | 12 | 21.9 | 27.3% | 0.27 | 77771111 | 77777171 |
| (c) goal-centre+goal-spec | 13 | 21.7 | 37.3% | 0.34 | 11114444 | 11111414 |
| (c) goal-centre+goal-spec | 14 | 20.4 | 97.3% | 0.46 | 22224444 | 44444444 |
| (c) goal-centre+goal-spec | 15 | 19.0 | 87.7% | 0.28 | 77000007 | 07000000 |
| (d) hand-goal-centre+learned-spec | 0 | 19.6 | 100.0% | 0.97 | 11234567 | 00000000 |
| (d) hand-goal-centre+learned-spec | 1 | 22.5 | 100.0% | 0.98 | 11234567 | 00000000 |
| (d) hand-goal-centre+learned-spec | 2 | 21.0 | 100.0% | 0.99 | 11234567 | 00000000 |
| (d) hand-goal-centre+learned-spec | 3 | 17.8 | 100.0% | 0.98 | 11234567 | 00000000 |
| (d) hand-goal-centre+learned-spec | 4 | 24.1 | 100.0% | 0.99 | 11234567 | 00000000 |
| (d) hand-goal-centre+learned-spec | 5 | 23.6 | 100.0% | 0.98 | 11234567 | 00000000 |
| (d) hand-goal-centre+learned-spec | 6 | 23.3 | 100.0% | 0.99 | 11234567 | 00000000 |
| (d) hand-goal-centre+learned-spec | 7 | 22.7 | 100.0% | 0.97 | 11234567 | 00000000 |
| (d) hand-goal-centre+learned-spec | 8 | 22.6 | 100.0% | 0.98 | 11234567 | 00000000 |
| (d) hand-goal-centre+learned-spec | 9 | 24.2 | 100.0% | 0.98 | 11234567 | 00000000 |
| (d) hand-goal-centre+learned-spec | 10 | 15.4 | 100.0% | 1.00 | 11234567 | 00000000 |
| (d) hand-goal-centre+learned-spec | 11 | 22.7 | 100.0% | 0.99 | 11234567 | 00000000 |
| (d) hand-goal-centre+learned-spec | 12 | 17.2 | 100.0% | 0.97 | 11234567 | 00000000 |
| (d) hand-goal-centre+learned-spec | 13 | 23.7 | 100.0% | 0.97 | 11234567 | 00000000 |
| (d) hand-goal-centre+learned-spec | 14 | 22.2 | 100.0% | 0.96 | 11234567 | 00000000 |
| (d) hand-goal-centre+learned-spec | 15 | 23.5 | 100.0% | 0.97 | 11234567 | 00000000 |

Situation s = 2 x direction + far of the receiver's nearest own-colour food (R6's hand code, `HandCodedCentre`).

Training settings: `{"radius": 3, "generations": 10000, "population": 64, "reps": 1, "sigma": 0.05, "lr": 0.03, "weight_decay": 0.005, "spec_hidden": 24, "centre_hidden": 24, "eval_every": 50, "val_reps": 2, "max_steps": 300}`.

Reproduce: `python scripts/r7d_legibility.py --from-json runs:r7b3-train-* runs:r7b4-train-* runs:r7b5-train-* runs:r7c-train-*`.
