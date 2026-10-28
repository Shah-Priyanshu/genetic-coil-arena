# R12b: a newcomer that reports what its errors are about (goal G9b)

Generated 2026-10-27 23:26. Joined rule (pinned before any fresh or test run): `a_gate45_cap50`. State hash of the old team before and after joining and running: `65a5c5926b2b905c6e36a94756ade66702991c6fda873455afd7d9d5a6691bc6` / `65a5c5926b2b905c6e36a94756ade66702991c6fda873455afd7d9d5a6691bc6` (identical).

Built locally from the `r12b-team` job artifacts on gpu-results (pinned commit 862865c6); the job wrote the same report into its run directory.

The RE3 and RE2 SS/OB test splits were already used by G9 (RE3 also by G4d and G4e), and the attribution idea came from G9's test-split failure analysis, so those two rows are re-scored on reused data and are weaker evidence than the Train Ticket row.

## Dev table (pooled RE2 SS/OB repetition 1 + RE3 repetition 1; chooses the rule)

| Rule | Pooled top-1 | RE2 top-1 | RE3 top-1 | RE2 joint | Log part used |
|---|---|---|---|---|---|
| `metric` (old team) | 78.4 | 86.7 | 60.7 | 66.7 | n/a |
| `g9_gate45_cap50` (G9 rule, reference, not selectable) | 84.1 | 81.7 | 89.3 | 63.3 | 39.8 |
| `attr_alone` (newcomer v2 alone, not selectable) | 64.8 | 58.3 | 78.6 | 45.0 | 62.5 |
| `a_rr_w1` | 68.2 | 58.3 | 89.3 | 45.0 | 62.5 |
| `a_gate40` | 81.8 | 80.0 | 85.7 | 63.3 | 37.5 |
| `a_gate45` | 83.0 | 80.0 | 89.3 | 63.3 | 42.0 |
| `a_gate50` | 68.2 | 58.3 | 89.3 | 45.0 | 61.4 |
| `a_cap50` | 79.5 | 75.0 | 89.3 | 58.3 | 48.9 |
| `a_gate40_cap50` | 83.0 | 81.7 | 85.7 | 63.3 | 35.2 |
| `a_gate45_cap50` (chosen) | 84.1 | 81.7 | 89.3 | 63.3 | 39.8 |
| `a_gate45_cap100` | 84.1 | 81.7 | 89.3 | 63.3 | 39.8 |

## Verdict: **KILL** (RE2 SS/OB top-1 significantly below the old team)

| Criterion | Target | Measured (points, paired 95% CI) | Result |
|---|---|---|---|
| (1) RE2 Train Ticket (fresh) top-1, joined - old | lower CI bound > -5 points | -4.4 [-10.0, +0.0] points (n=90) | not met |
| (2) RE3 test (reused) top-1, joined - old | >= +10 points, CI excludes 0 | +22.6 [+12.9, +33.9] points (n=62) | PASS |
| (3) RE2 SS/OB test (reused) top-1, joined - old | lower CI bound > -5 points | -3.3 [-6.7, -0.8] points (n=120) | not met |
| (3) RE2 SS/OB test (reused) joint, joined - old | lower CI bound > -5 points | -3.3 [-6.7, -0.8] points (n=120) | not met |

## RE2 Train Ticket (fresh, never ranked in a test), n = 90

| System | Top-1 | Top-3 | Joint | This - old (top-1, paired 95% CI) | Fixed / broken |
|---|---|---|---|---|---|
| old team | 78.9 | 91.1 | n/a | - | - |
| joined v2 (`a_gate45_cap50`) | 74.4 | 91.1 | n/a | -4.4 [-10.0, +0.0] | 1 / 5 |
| G9 joined team (`g9_gate45_cap50`, v1 counts) | 75.6 | 92.2 | n/a | -3.3 [-7.8, +1.1] | 1 / 4 |
| newcomer v2 alone (`n_attr`) | 58.9 | 80.0 | n/a | -20.0 [-28.9, -11.1] | 1 / 19 |
| BARO on metrics | 78.9 | 91.1 | - | joined - this: -4.4 [-10.0, +0.0] | - |
| G9 newcomer alone (new_ext) | 6.7 | 14.4 | - | joined - this: +67.8 [+56.7, +78.9] | - |
| RE3-only fusion (baro 0.5 + new_ext 0.5) | 62.2 | 92.2 | - | joined - this: +12.2 [+5.6, +20.0] | - |

The log part took part in 11.1% of incidents and changed the top-1 in 8.9%. Attribution moved a count in 31.1% of incidents (162 of 534 new templates were credited to another service).

| System | n | Old top-1 | Joined v2 top-1 | G9 rule top-1 | Fixed / broken |
|---|---|---|---|---|---|
| TT | 90 | 78.9 | 74.4 | 75.6 | 1 / 5 |

| Fault | n | Old top-1 | Joined v2 top-1 | G9 rule top-1 | Fixed / broken |
|---|---|---|---|---|---|
| cpu | 15 | 100.0 | 73.3 | 80.0 | 0 / 4 |
| delay | 15 | 40.0 | 33.3 | 33.3 | 0 / 1 |
| disk | 15 | 100.0 | 100.0 | 100.0 | 0 / 0 |
| loss | 15 | 40.0 | 46.7 | 46.7 | 1 / 0 |
| mem | 15 | 100.0 | 100.0 | 100.0 | 0 / 0 |
| socket | 15 | 93.3 | 93.3 | 93.3 | 0 / 0 |

## RE3 test, repetitions 2-4 (REUSED: G4d, G4e and G9 used it), n = 62

| System | Top-1 | Top-3 | Joint | This - old (top-1, paired 95% CI) | Fixed / broken |
|---|---|---|---|---|---|
| old team | 56.5 | 90.3 | n/a | - | - |
| joined v2 (`a_gate45_cap50`) | 79.0 | 88.7 | n/a | +22.6 [+12.9, +33.9] | 14 / 0 |
| G9 joined team (`g9_gate45_cap50`, v1 counts) | 79.0 | 88.7 | n/a | +22.6 [+12.9, +33.9] | 14 / 0 |
| newcomer v2 alone (`n_attr`) | 79.0 | 98.4 | n/a | +22.6 [+6.5, +38.7] | 22 / 8 |
| BARO on metrics | 56.5 | 91.9 | - | joined - this: +22.6 [+12.9, +33.9] | - |
| G9 newcomer alone (new_ext) | 77.4 | 91.9 | - | joined - this: +1.6 [-11.3, +14.5] | - |
| RE3-only fusion (baro 0.5 + new_ext 0.5) | 77.4 | 93.5 | - | joined - this: +1.6 [-6.5, +9.7] | - |

The log part took part in 85.5% of incidents and changed the top-1 in 27.4%. Attribution moved a count in 25.8% of incidents (19 of 2260 new templates were credited to another service).

| System | n | Old top-1 | Joined v2 top-1 | G9 rule top-1 | Fixed / broken |
|---|---|---|---|---|---|
| RE3-OB | 20 | 60.0 | 70.0 | 70.0 | 2 / 0 |
| RE3-SS | 21 | 71.4 | 90.5 | 90.5 | 4 / 0 |
| RE3-TT | 21 | 38.1 | 76.2 | 76.2 | 8 / 0 |

## RE2 SS/OB test, repetitions 2-3 (REUSED: G9 used it), n = 120

| System | Top-1 | Top-3 | Joint | This - old (top-1, paired 95% CI) | Fixed / broken |
|---|---|---|---|---|---|
| old team | 88.3 | 99.2 | 68.3 | - | - |
| joined v2 (`a_gate45_cap50`) | 85.0 | 100.0 | 65.0 | -3.3 [-6.7, -0.8] | 0 / 4 |
| G9 joined team (`g9_gate45_cap50`, v1 counts) | 85.0 | 100.0 | 65.0 | -3.3 [-6.7, -0.8] | 0 / 4 |
| newcomer v2 alone (`n_attr`) | 72.5 | 97.5 | 54.2 | -15.8 [-22.5, -9.2] | 0 / 19 |
| BARO on metrics | 88.3 | 99.2 | - | joined - this: -3.3 [-6.7, -0.8] | - |
| G9 newcomer alone (new_ext) | 5.8 | 22.5 | - | joined - this: +79.2 [+71.7, +85.8] | - |
| RE3-only fusion (baro 0.5 + new_ext 0.5) | 75.0 | 100.0 | - | joined - this: +10.0 [+5.0, +15.8] | - |

The log part took part in 15.0% of incidents and changed the top-1 in 4.2%. Attribution moved a count in 5.0% of incidents (15 of 165 new templates were credited to another service).

| System | n | Old top-1 | Joined v2 top-1 | G9 rule top-1 | Fixed / broken |
|---|---|---|---|---|---|
| OB | 60 | 83.3 | 80.0 | 80.0 | 0 / 2 |
| SS | 60 | 93.3 | 90.0 | 90.0 | 0 / 2 |

| Fault | n | Old top-1 | Joined v2 top-1 | G9 rule top-1 | Fixed / broken |
|---|---|---|---|---|---|
| cpu | 20 | 95.0 | 95.0 | 95.0 | 0 / 0 |
| delay | 20 | 85.0 | 85.0 | 85.0 | 0 / 0 |
| disk | 20 | 95.0 | 95.0 | 95.0 | 0 / 0 |
| loss | 20 | 60.0 | 40.0 | 40.0 | 0 / 4 |
| mem | 20 | 100.0 | 100.0 | 100.0 | 0 / 0 |
| socket | 20 | 95.0 | 95.0 | 95.0 | 0 / 0 |

## Analysis

<!-- analysis:start -->
**G9b is a kill. Attributing errors to the service they name does not remove the newcomer's cost on old incidents.**

**The reused RE2 Sock Shop / Online Boutique split is unchanged from G9.**
- Joined v2 − old: −3.3 [−6.7, −0.8] top-1 and joint, with the same 4 breaks. All are packet-loss faults.
- Attribution moved a count in only 5% of these incidents (15 of 165 new templates), and in none of the four loss incidents.
- The callers' error lines on packet loss do not name the callee in any form the matcher recognises: host:port, URL, DNS label or `UnknownHostException`. The idea from G9's analysis does not hold in these logs.

**On the fresh Train Ticket incidents (90, never ranked in a test before), the cost appears again at a similar size.**

| Rule | Top-1 vs old team (78.9) | Change [95% CI] | Fixes | Breaks |
|---|---|---|---|---|
| v2, attributed counts | 74.4 | −4.4 [−10.0, +0.0] | 1 | 5 |
| G9 rule, plain counts | 75.6 | −3.3 [−7.8, +1.1] | 1 | 4 |

- Neither loss is significant with 90 incidents, so this criterion is "not met" rather than a kill on its own. The direction and size match G9's −3.3 on Sock Shop / Online Boutique.
- **The breaks differ in kind:** 4 of v2's 5 are CPU faults (CPU top-1 falls from 100% to 73.3%).
  - Three of them are CPU faults on `ts-auth-service`. The gate let the logs in, and the newcomer pointed at `ts-travel-service` twice and at `ts-auth-mongo` once.
  - In Train Ticket, other services log new errors during a fault too, so a weak metric anomaly plus scattered new errors misleads the same way packet loss does on the shops.
- **Attribution was active on Train Ticket and made things slightly worse.** It moved a count in 31% of incidents: 162 of 534 new templates were credited to another service, because the Java services log the hosts they call. The result was −4.4 against −3.3.
- **On RE3 attribution changed nothing:** 19 of 2,260 templates moved, and the results are identical to G9 (+22.6, 14 fixes, 0 breaks).

**A caution on "newcomer v2 alone".** That row breaks ties by the metric score (its rule is `rr(n_attr)` with z only as the tie-break). Where no service has attributed novelty it falls back on the metric ranking. Its standalone numbers (72.5% on RE2 Sock Shop / Online Boutique, 58.9% on Train Ticket) therefore cannot be compared with the v1 newcomer alone (5.8%, 6.7%), whose ties go by service order.

**What G9 and G9b say together.** A code newcomer joining through a rule centre:
- adds about 23 points on the incident class it was built for (RE3 code-level faults);
- costs the old class 3 to 4 points, on two separate data sets, the reused RE2 Sock Shop / Online Boutique test and fresh RE2 Train Ticket.

The cost comes from the same place each time: incidents whose metric evidence is weak, where the gate lets the log part in and new errors appear in services other than the root.

A cleverer count does not fix that. Making error lines point at the service they mention did not help where it was needed and hurt where it applied. What is missing is a signal that separates "the logs know the root" (code faults) from "the logs echo a fault elsewhere" (resource and network faults), and the reports as built do not carry it.

Two options remain, both untested here:
- An input the desk already has in practice: the alert's own type (error-rate against latency or resource alerts).
- A learned centre over richer per-service features. That needs more labelled incidents than the 88 on dev.

**The test splits were reused.** The RE2 Sock Shop / Online Boutique and RE3 test splits are reused here (G9; and G4d, G4e for RE3). The attribution idea came from G9's test-split failure analysis. The fresh Train Ticket incidents are the clean evidence in this report, and they show the same cost.
<!-- analysis:end -->
