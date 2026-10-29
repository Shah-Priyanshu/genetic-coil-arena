# R17: does a leader that reads the mission plan the search better?

**Room: PASS.** Tuning the plan per mission beats the one blind plan by +22.6 ± 6.6 points (bar ≥ +5, CI above 0).

**Good sign: PASS.** The conditioned planner beats the one blind plan by +18.0 ± 6.1 points (bar ≥ +3, CI above 0).

40 held-out missions × 2000 episodes; score = found − 2 × wrong find, in points; differences are paired by mission (95% CI).

| System | Runs | Score | Found | Wrong find | Lanes flown | vs blind |
|---|---|---|---|---|---|---|
| OpenSwarm default | 1 | -69.8 | 33.9% | 51.9% | 33% | -116.4 ± 20.1 |
| hand two-tier | 1 | 38.2 | 46.6% | 4.2% | 68% | -8.3 ± 4.5 |
| blind | 16 | 46.6 | 53.7% | 3.6% | 57% | — |
| conditioned | 16 | 64.6 | 69.0% | 2.2% | 61% | +18.0 ± 6.1 |
| per-mission tuned | 1 | 69.1 | 74.0% | 2.4% | 60% | +22.6 ± 6.6 |

Plans chosen (run 0) on telling missions:

- **smallest area** (66 × 110 m, 6 drones, px50 8, 0.0111 false/look):
  - OpenSwarm default: sweep 5 m at 0.70
  - hand two-tier: sweep 8 m at 0.50, 2 confirm at 4 m, 3 looks at 0.70
  - blind: sweep 5 m at 0.58, 1 confirm at 4 m, 4 looks at 0.73
  - conditioned: sweep 5 m at 0.57, 1 confirm at 7 m, 10 looks at 0.73
  - per-mission tuned: sweep 8 m at 0.55, 1 confirm at 5 m, 5 looks at 0.75
- **largest area** (206 × 229 m, 3 drones, px50 13, 0.0107 false/look):
  - OpenSwarm default: sweep 5 m at 0.70
  - hand two-tier: sweep 8 m at 0.50, 1 confirm at 4 m, 3 looks at 0.70
  - blind: sweep 5 m at 0.58, 1 confirm at 4 m, 4 looks at 0.73
  - conditioned: sweep 11 m at 0.48, 1 confirm at 4 m, 7 looks at 0.88
  - per-mission tuned: sweep 7 m at 0.50, 1 confirm at 4 m, 3 looks at 0.79
- **best detector** (226 × 183 m, 4 drones, px50 7, 0.0007 false/look):
  - OpenSwarm default: sweep 5 m at 0.70
  - hand two-tier: sweep 8 m at 0.50, 1 confirm at 4 m, 3 looks at 0.70
  - blind: sweep 5 m at 0.58, 1 confirm at 4 m, 4 looks at 0.73
  - conditioned: sweep 21 m at 0.30, 1 confirm at 8 m, 10 looks at 0.66
  - per-mission tuned: sweep 18 m at 0.31, 1 confirm at 9 m, 3 looks at 0.47
- **worst detector** (205 × 83 m, 4 drones, px50 19, 0.0083 false/look):
  - OpenSwarm default: sweep 5 m at 0.70
  - hand two-tier: sweep 8 m at 0.50, 1 confirm at 4 m, 3 looks at 0.70
  - blind: sweep 5 m at 0.58, 1 confirm at 4 m, 4 looks at 0.73
  - conditioned: sweep 5 m at 0.40, 1 confirm at 3 m, 3 looks at 0.76
  - per-mission tuned: sweep 5 m at 0.48, 1 confirm at 4 m, 5 looks at 0.72
- **fewest false alarms** (220 × 153 m, 8 drones, px50 16, 0.0005 false/look):
  - OpenSwarm default: sweep 5 m at 0.70
  - hand two-tier: sweep 8 m at 0.50, 2 confirm at 4 m, 3 looks at 0.70
  - blind: sweep 5 m at 0.58, 2 confirm at 4 m, 4 looks at 0.73
  - conditioned: sweep 8 m at 0.31, 1 confirm at 8 m, 9 looks at 0.43
  - per-mission tuned: sweep 5 m at 0.33, 1 confirm at 7 m, 8 looks at 0.44
- **most false alarms** (128 × 82 m, 8 drones, px50 18, 0.0298 false/look):
  - OpenSwarm default: sweep 5 m at 0.70
  - hand two-tier: sweep 8 m at 0.50, 2 confirm at 4 m, 3 looks at 0.70
  - blind: sweep 5 m at 0.58, 2 confirm at 4 m, 4 looks at 0.73
  - conditioned: sweep 3 m at 0.75, 1 confirm at 3 m, 10 looks at 0.82
  - per-mission tuned: sweep 3 m at 0.70, 2 confirm at 3 m, 6 looks at 0.84

Reproduce: `python scripts/r17_mission.py report --from runs:r17-*`.
