# R12: growth on real incidents (goals G9 and G10)

Generated 2026-10-27 22:53. Command: `python scripts/r12_growth.py report`.

Built locally from the job artifacts of `r12-team`, `r12-o14-sealed`, `r12-gemma-sealed` and `r12-keyword-sealed` on gpu-results (pinned commit fc1832a9, with the R12 code of 0e1a88c5), the same inputs as the runner's `r12-zz-aggregate`.

## G9: the log-novelty specialist joins the incident team

Joined rule (pinned before the test run): `gate45_cap50`. State hash of the old team before and after joining and running: `65a5c5926b2b905c6e36a94756ade66702991c6fda873455afd7d9d5a6691bc6` / `65a5c5926b2b905c6e36a94756ade66702991c6fda873455afd7d9d5a6691bc6` (identical).

### Dev table (pooled RE2 SS/OB repetition 1 + RE3 repetition 1; chooses the rule)

| Rule | Pooled top-1 | RE2 top-1 | RE3 top-1 | RE2 joint | Log part used |
|---|---|---|---|---|---|
| `metric` (old team) | 78.4 | 86.7 | 60.7 | 66.7 | - |
| `rr_w0.5` | 79.5 | 86.7 | 64.3 | 66.7 | 62.5 |
| `rr_w1` | 68.2 | 58.3 | 89.3 | 45.0 | 62.5 |
| `rr_w2` | 64.8 | 58.3 | 78.6 | 45.0 | 62.5 |
| `gate40` | 81.8 | 80.0 | 85.7 | 63.3 | 37.5 |
| `gate45` | 83.0 | 80.0 | 89.3 | 63.3 | 42.0 |
| `cap20` | 80.7 | 76.7 | 89.3 | 60.0 | 47.7 |
| `cap50` | 79.5 | 75.0 | 89.3 | 58.3 | 48.9 |
| `gate45_cap50` (chosen) | 84.1 | 81.7 | 89.3 | 63.3 | 39.8 |
| `gate40_cap50` | 83.0 | 81.7 | 85.7 | 63.3 | 35.2 |

The dev incidents of RE2 repetition 1 are in the fault specialist's training data, so the dev joint accuracy is optimistic; the ranking does not depend on it.

### Verdict on the test split: **KILL** (RE2 top-1 significantly below the old team; RE2 joint significantly below the old team)

| Criterion | Target | Measured | Result |
|---|---|---|---|
| RE3 top-1, joined - old | >= +10 points, CI excludes 0 | +22.6 [+12.9, +33.9] points (n=62) | PASS |
| RE2 top-1, joined - old | lower CI bound > -5 points | -3.3 [-6.7, -0.8] points (n=120) | not met |
| RE2 joint, joined - old | lower CI bound > -5 points | -3.3 [-6.7, -0.8] points (n=120) | not met |

### RE2 SS/OB (repetitions 2-3), n = 120

| System | Top-1 | Top-3 | Joint | Joined - this (top-1, paired 95% CI) |
|---|---|---|---|---|
| old team | 88.3 | 99.2 | 68.3 | joined -3.3 [-6.7, -0.8] |
| joined team (`gate45_cap50`) | 85.0 | 100.0 | 65.0 | - |
| BARO on metrics | 88.3 | 99.2 | - | joined -3.3 [-6.7, -0.8] |
| newcomer alone (new_ext) | 5.8 | 22.5 | - | joined +79.2 [+71.7, +85.8] |
| RE3-only fusion (baro 0.5 + new_ext 0.5) | 75.0 | 100.0 | - | joined +10.0 [+5.0, +15.8] |

The log part took part in 15.0% of incidents and changed the top-1 in 4.2% (0 fixed, 4 broken).

| System | n | Old top-1 | Joined top-1 | Log part used (%) |
|---|---|---|---|---|
| OB | 60 | 83.3 | 80.0 | 15.0 |
| SS | 60 | 93.3 | 90.0 | 15.0 |

### RE3 (repetitions 2-4), n = 62

| System | Top-1 | Top-3 | Joint | Joined - this (top-1, paired 95% CI) |
|---|---|---|---|---|
| old team | 56.5 | 90.3 | n/a | joined +22.6 [+12.9, +33.9] |
| joined team (`gate45_cap50`) | 79.0 | 88.7 | n/a | - |
| BARO on metrics | 56.5 | 91.9 | - | joined +22.6 [+12.9, +33.9] |
| newcomer alone (new_ext) | 77.4 | 91.9 | - | joined +1.6 [-11.3, +14.5] |
| RE3-only fusion (baro 0.5 + new_ext 0.5) | 77.4 | 93.5 | - | joined +1.6 [-6.5, +9.7] |

The log part took part in 85.5% of incidents and changed the top-1 in 27.4% (14 fixed, 0 broken).

| System | n | Old top-1 | Joined top-1 | Log part used (%) |
|---|---|---|---|---|
| RE3-OB | 20 | 60.0 | 70.0 | 55.0 |
| RE3-SS | 21 | 71.4 | 90.5 | 100.0 |
| RE3-TT | 21 | 38.1 | 76.2 | 100.0 |

## G10: the desk grows to Train Ticket and the RE3 incidents

- **Sealed set:** 107 hand-written requests (`experiments/data/r12_sealed_test.jsonl`, SHA-256 `a0285faf8d99d69e1e732ad02a0bca8ca205c2ef7f51606bd11e322dc614c1ea`), plus the R7a sealed spec and clarify items for the routing check (SHA-256 `ca6ff07b5fb66ef21c288ba9a82d5e7763992fe2142b1a582baf597e542b5e01`). Each translator ran once.
- **Catalogue:** 182 test incidents (RE2 SS/OB repetitions 2-3 as `INC-1001`..`INC-1120`, RE3 repetitions 2-4 as `INC-2001`..`INC-2062`), fingerprint `12e4ce9fd1d393ee7c01092ae35572b28b4d73d0e33357327075e186a9bd8952`; desk clock Wednesday 2024-12-11 09:00 UTC.
- **Prompt audit:** v3 differs from v2 in 4 system-prompt lines (the clock line, and the aliases and services paragraph, the system list and the refusal scope rendered from the systems table) and 1 few-shot date (the 'yesterday' example, of 19); the router's incident line is rendered from the table too, all else is the v2 text: True.
- **Team:** the G9 joined team (rule `gate45_cap50`) over the G4c trained metric fault specialist (config {'hidden': 0, 'l2': 1.0, 'epochs': 400}, 150 training incidents).

### Verdict

| Translator | G10 verdict | Exact RE3 | Exact RE2 | Routing | NL - gold, RE3 | NL - gold, RE2 | Round trip | Fidelity |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-coder:14b (refusal policy) | **NOT PASSED** | 87.9% (58/66) | 88.0% (22/25) | 100.0% (252/252) | +0.0 [+0.0, +0.0] | -4.0 [-12.0, +0.0] | 95.6% (86/90) | 100.0% |
| gemma4:e4b (refusal policy) | **KILL** | 81.8% (54/66) | 92.0% (23/25) | 99.6% (251/252) | -9.1 [-16.7, -3.0] | -4.0 [-12.0, +0.0] | 86.4% (70/81) | 99.5% |
| keyword/regex parser | **KILL** | 83.3% (55/66) | 92.0% (23/25) | 98.0% (247/252) | -7.6 [-13.6, -1.5] | -4.0 [-12.0, +0.0] | n/a | n/a |

### Per translator

### qwen2.5-coder:14b (refusal policy)

**NOT PASSED**

| Criterion | Target | Measured | Result |
|---|---|---|---|
| exact match, RE3 (new system and incidents) | >= 90% (kill < 85%) | 87.9% (n=66, 95% CI 77.9%-93.7%) | not met |
| exact match, RE2 (regression) | >= 90% (kill < 85%) | 88.0% (n=25, 95% CI 70.0%-95.8%) | not met |
| domain routing | >= 98% | 100.0% (n=252, 95% CI 98.5%-100.0%) | PASS |
| task score (root top-1), RE3, NL vs gold | lower CI bound of NL - gold > -5 points | NL 74.2% vs gold 74.2%; diff +0.0 [+0.0, +0.0] points (n=66) | PASS |
| task score (joint), RE2, NL vs gold | lower CI bound of NL - gold > -5 points | NL 60.0% vs gold 64.0%; diff -4.0 [-12.0, +0.0] points (n=25) | not met |
| round trip | >= 90% | 95.6% (n=90, 95% CI 89.1%-98.3%) | PASS |
| summary field fidelity | >= 90% of facts | 100.0% of 669 facts; all facts in 100.0% of 90 summaries | PASS |
| warm end-to-end p95 / peak VRAM (reported) | - | 4.41 s, 10.5 GB | - |

Routing by desk: incident 100.0% (97/97); plant 100.0% (76/76); squad 100.0% (79/79).

Exact match by system: online-boutique (re2) 76.9% (10/13); online-boutique (re3) 87.5% (14/16); sock-shop (re2) 100.0% (12/12); sock-shop (re3) 83.3% (15/18); train-ticket (re3) 90.6% (29/32).

Final decision on all 107 items: 99.1% (106/107). clarify: 100.0% (6/6) {'clarify': 6}. refuse: 100.0% (10/10) {'refused': 10}. spec: 98.9% (90/91) {'ok': 90, 'refused': 1}.

| Category | Exact match, all spec items | Decision (all kinds) |
|---|---|---|
| clarify | - | 100.0% (6/6) |
| combo | 66.7% (4/6) | 83.3% (5/6) |
| exclude | 100.0% (11/11) | 100.0% (11/11) |
| fault | 100.0% (5/5) | 100.0% (5/5) |
| partial | 59.1% (13/22) | 100.0% (22/22) |
| refuse | - | 100.0% (10/10) |
| relative | 100.0% (6/6) | 100.0% (6/6) |
| ticket | 100.0% (21/21) | 100.0% (21/21) |
| time | 100.0% (12/12) | 100.0% (12/12) |
| topk | 100.0% (8/8) | 100.0% (8/8) |

Fact accuracy: at 100.0%, exclusions 100.0%, fault_probability 100.0%, fault_type 100.0%, onset 100.0%, suspects 100.0%, system 100.0%, ticket 100.0%.

#### qwen2.5-coder:14b (refusal policy): failure examples

translation (11 shown):

- `r12-sealed-023` category=partial; style=plain; text=Online boutique on Nov 22 at 11:59, what went wrong?; gold={'at': '--11-22T11:59', 'exclude_services': [], 'system': 'online-boutique', 'top_k': 3, 'want_fault_type': True}; pred={'system': 'online-boutique', 'at': '2024-11-22T11:59', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; status=ok; routed=incident
- `r12-sealed-025` category=partial; style=terse; text=sock shop incident, nov 22nd, 7 am sharp; gold={'at': '--11-22T07:00', 'exclude_services': [], 'system': 'sock-shop', 'top_k': 3, 'want_fault_type': True}; pred={'system': 'sock-shop', 'at': '2024-11-22T07:00', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; status=ok; routed=incident
- `r12-sealed-026` category=partial; style=formal; text=Kindly investigate the Sock Shop incident of 24 November at 13:08.; gold={'at': '--11-24T13:08', 'exclude_services': [], 'system': 'sock-shop', 'top_k': 3, 'want_fault_type': True}; pred={'system': 'sock-shop', 'at': '2024-11-24T13:08', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; status=ok; routed=incident
- `r12-sealed-027` category=partial; style=plain; text=Train ticket, Dec 8 at 11:36, find the culprit.; gold={'at': '--12-08T11:36', 'exclude_services': [], 'system': 'train-ticket', 'top_k': 3, 'want_fault_type': True}; pred={'system': 'train-ticket', 'at': '2024-12-08T11:36', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; status=ok; routed=incident
- `r12-sealed-028` category=partial; style=typo; text=train ticket isue of 8 decemeber around 16:19; gold={'at': '--12-08T16:19', 'exclude_services': [], 'system': 'train-ticket', 'top_k': 3, 'want_fault_type': True}; pred={'system': 'train-ticket', 'at': '2024-12-08T16:19', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; status=ok; routed=incident
- `r12-sealed-029` category=partial; style=words; text=Ten December, train ticket system, twelve thirty-eight.; gold={'at': '--12-10T12:38', 'exclude_services': [], 'system': 'train-ticket', 'top_k': 3, 'want_fault_type': True}; pred={'system': 'train-ticket', 'at': '2024-12-10T12:38', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; status=ok; routed=incident
- `r12-sealed-030` category=partial; style=plain; text=Sock shop incident Nov 23 at 9:34 am.; gold={'at': '--11-23T09:34', 'exclude_services': [], 'system': 'sock-shop', 'top_k': 3, 'want_fault_type': True}; pred={'system': 'sock-shop', 'at': '2024-11-23T09:34', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; status=ok; routed=incident
- `r12-sealed-063` category=combo; style=chatty; text=Boutique on Nov 22 at 14:07, could you give me five suspects and leave out adservice?; gold={'at': '--11-22T14:07', 'exclude_services': ['adservice'], 'system': 'online-boutique', 'top_k': 5, 'want_fault_type': True}; pred={'system': 'online-boutique', 'at': '2024-11-22T14:07', 'top_k': 5, 'exclude_services': ['adservice'], 'want_fault_type': True}; status=ok; routed=incident
- `r12-sealed-077` category=partial; style=plain; text=Boutique, February 20th, 23:49.; gold={'at': '--02-20T23:49', 'exclude_services': [], 'system': 'online-boutique', 'top_k': 3, 'want_fault_type': True}; pred={'system': 'online-boutique', 'at': '---20T23:49', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; status=ok; routed=incident
- `r12-sealed-080` category=partial; style=chatty; text=The boutique on the 19th around 12:55 - what was that?; gold={'at': '---19T12:55', 'exclude_services': [], 'system': 'online-boutique', 'top_k': 3, 'want_fault_type': True}; pred={'system': 'online-boutique', 'at': '--01-19T12:55', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; status=ok; routed=incident
- `r12-sealed-089` category=combo; style=plain; text=Online boutique on the 19th at 21:26, 4 suspects, leave out redis, no fault type.; gold={'at': '---19T21:26', 'exclude_services': ['redis'], 'system': 'online-boutique', 'top_k': 4, 'want_fault_type': False}; pred={'system': 'online-boutique', 'at': '--01-19T21:26', 'top_k': 4, 'exclude_services': ['redis', 'want_fault_type'], 'want_fault_type': True}; status=refused; reason=online-boutique has no service named want_fault_type; routed=incident

round_trip (4 shown):

- `r12-sealed-054` summary=Investigation of the online-boutique incident, requested by ticket INC-2025, top 3 suspects requested. The suspected services, most likely first, are emailservice, cartservice, recommendationservice; fault type delay (p...; recon_status=ok; recon_params={'system': 'online-boutique', 'ticket': 'INC-2025', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; gold={'exclude_services': ['frontend', 'frontend-external'], 'system': 'online-boutique', 'ticket': 'INC-2025', 'top_k': 3, 'want_fault_type': True}
- `r12-sealed-055` summary=Investigation of the train-ticket incident, requested by ticket INC-2045, top 3 suspects requested. The suspected services, most likely first, are ts-auth-service, ts-order-service, ts-order-other-service; fault type lo...; recon_status=ok; recon_params={'system': 'train-ticket', 'ticket': 'INC-2045', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; gold={'exclude_services': ['ts-ui-dashboard'], 'system': 'train-ticket', 'ticket': 'INC-2045', 'top_k': 3, 'want_fault_type': True}
- `r12-sealed-056` summary=Investigation of the train-ticket incident, requested by ticket INC-2054, top 3 suspects requested. The suspected services, most likely first, are ts-route-service, ts-inside-payment-service, ts-admin-travel-service; fa...; recon_status=ok; recon_params={'system': 'train-ticket', 'ticket': 'INC-2054', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; gold={'exclude_services': ['ts-route-mongo', 'ts-travel-mongo'], 'system': 'train-ticket', 'ticket': 'INC-2054', 'top_k': 3, 'want_fault_type': True}
- `r12-sealed-084` summary=Investigation of the online-boutique incident, requested by ticket INC-1040, top 3 suspects requested. The suspected services, most likely first, are currencyservice, frontend, checkoutservice; fault type socket (probab...; recon_status=ok; recon_params={'system': 'online-boutique', 'ticket': 'INC-1040', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; gold={'exclude_services': ['redis'], 'system': 'online-boutique', 'ticket': 'INC-1040', 'top_k': 3, 'want_fault_type': True}

### gemma4:e4b (refusal policy)

**KILL** (RE3 exact match 81.8% < 85%; RE3 NL task score significantly below the gold run)

| Criterion | Target | Measured | Result |
|---|---|---|---|
| exact match, RE3 (new system and incidents) | >= 90% (kill < 85%) | 81.8% (n=66, 95% CI 70.9%-89.3%) | not met |
| exact match, RE2 (regression) | >= 90% (kill < 85%) | 92.0% (n=25, 95% CI 75.0%-97.8%) | PASS |
| domain routing | >= 98% | 99.6% (n=252, 95% CI 97.8%-99.9%) | PASS |
| task score (root top-1), RE3, NL vs gold | lower CI bound of NL - gold > -5 points | NL 65.2% vs gold 74.2%; diff -9.1 [-16.7, -3.0] points (n=66) | not met |
| task score (joint), RE2, NL vs gold | lower CI bound of NL - gold > -5 points | NL 60.0% vs gold 64.0%; diff -4.0 [-12.0, +0.0] points (n=25) | not met |
| round trip | >= 90% | 86.4% (n=81, 95% CI 77.3%-92.2%) | not met |
| summary field fidelity | >= 90% of facts | 99.5% of 598 facts; all facts in 96.3% of 81 summaries | PASS |
| warm end-to-end p95 / peak VRAM (reported) | - | 2.47 s, 5.1 GB | - |

Routing by desk: incident 100.0% (97/97); plant 98.7% (75/76); squad 100.0% (79/79).

Exact match by system: online-boutique (re2) 84.6% (11/13); online-boutique (re3) 81.2% (13/16); sock-shop (re2) 100.0% (12/12); sock-shop (re3) 77.8% (14/18); train-ticket (re3) 84.4% (27/32).

Final decision on all 107 items: 89.7% (96/107). clarify: 100.0% (6/6) {'clarify': 6}. refuse: 90.0% (9/10) {'clarify': 1, 'refused': 9}. spec: 89.0% (81/91) {'clarify': 4, 'ok': 81, 'refused': 6}.

| Category | Exact match, all spec items | Decision (all kinds) |
|---|---|---|
| clarify | - | 100.0% (6/6) |
| combo | 66.7% (4/6) | 83.3% (5/6) |
| exclude | 100.0% (11/11) | 100.0% (11/11) |
| fault | 100.0% (5/5) | 100.0% (5/5) |
| partial | 54.5% (12/22) | 68.2% (15/22) |
| refuse | - | 90.0% (9/10) |
| relative | 83.3% (5/6) | 83.3% (5/6) |
| ticket | 95.2% (20/21) | 95.2% (20/21) |
| time | 100.0% (12/12) | 100.0% (12/12) |
| topk | 100.0% (8/8) | 100.0% (8/8) |

Fact accuracy: at 100.0%, exclusions 96.3%, fault_probability 100.0%, fault_type 100.0%, onset 100.0%, suspects 100.0%, system 100.0%, ticket 100.0%.

#### gemma4:e4b (refusal policy): failure examples

translation (12 shown):

- `r12-sealed-005` category=ticket; style=words; text=Ticket two thousand and twenty-four, the boutique.; gold={'exclude_services': [], 'system': 'online-boutique', 'ticket': 'INC-2024', 'top_k': 3, 'want_fault_type': True}; status=clarify; reason=Could you provide the specific ticket number or date for the boutique incident?; routed=incident
- `r12-sealed-023` category=partial; style=plain; text=Online boutique on Nov 22 at 11:59, what went wrong?; gold={'at': '--11-22T11:59', 'exclude_services': [], 'system': 'online-boutique', 'top_k': 3, 'want_fault_type': True}; pred={'system': 'online-boutique', 'at': '2024-11-22T23:59', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; status=refused; reason=no online-boutique incident within 10 minutes of 2024-11-22T23:59; routed=incident
- `r12-sealed-025` category=partial; style=terse; text=sock shop incident, nov 22nd, 7 am sharp; gold={'at': '--11-22T07:00', 'exclude_services': [], 'system': 'sock-shop', 'top_k': 3, 'want_fault_type': True}; status=refused; reason=the date format provided does not match the required schema for the 'at' parameter; routed=incident
- `r12-sealed-026` category=partial; style=formal; text=Kindly investigate the Sock Shop incident of 24 November at 13:08.; gold={'at': '--11-24T13:08', 'exclude_services': [], 'system': 'sock-shop', 'top_k': 3, 'want_fault_type': True}; pred={'system': 'sock-shop', 'at': '2024-11-24T13:08', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; status=ok; routed=incident
- `r12-sealed-027` category=partial; style=plain; text=Train ticket, Dec 8 at 11:36, find the culprit.; gold={'at': '--12-08T11:36', 'exclude_services': [], 'system': 'train-ticket', 'top_k': 3, 'want_fault_type': True}; pred={'system': 'train-ticket', 'at': '2024-12-08T11:36', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; status=ok; routed=incident
- `r12-sealed-028` category=partial; style=typo; text=train ticket isue of 8 decemeber around 16:19; gold={'at': '--12-08T16:19', 'exclude_services': [], 'system': 'train-ticket', 'top_k': 3, 'want_fault_type': True}; status=clarify; reason=Do you have a ticket number for this incident?; routed=incident
- `r12-sealed-029` category=partial; style=words; text=Ten December, train ticket system, twelve thirty-eight.; gold={'at': '--12-10T12:38', 'exclude_services': [], 'system': 'train-ticket', 'top_k': 3, 'want_fault_type': True}; status=refused; reason=the date format provided is ambiguous for the required 'at' parameter format; routed=incident
- `r12-sealed-030` category=partial; style=plain; text=Sock shop incident Nov 23 at 9:34 am.; gold={'at': '--11-23T09:34', 'exclude_services': [], 'system': 'sock-shop', 'top_k': 3, 'want_fault_type': True}; status=refused; reason=the date format provided does not match the required schema for the 'at' parameter; routed=incident
- `r12-sealed-033` category=partial; style=terse; text=shop 23rd 21:35; gold={'at': '---23T21:35', 'exclude_services': [], 'system': 'sock-shop', 'top_k': 3, 'want_fault_type': True}; status=clarify; reason=What is the ticket number for the incident on the sock-shop on the 23rd at 21:35?; routed=incident
- `r12-sealed-038` category=partial; style=keyvalue; text=system=train-ticket at=---08T15:44; gold={'at': '---08T15:44', 'exclude_services': [], 'system': 'train-ticket', 'top_k': 3, 'want_fault_type': True}; status=clarify; reason=Please provide a ticket number for the incident investigation.; routed=incident
- `r12-sealed-041` category=relative; style=question; text=What was wrong with the Train Ticket platform yesterday around 12:05?; gold={'at': '2024-12-10T12:05', 'exclude_services': [], 'system': 'train-ticket', 'top_k': 3, 'want_fault_type': True}; pred={'system': 'train-ticket', 'at': '2024-12-10T00:05', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; status=refused; reason=no train-ticket incident within 10 minutes of 2024-12-10T00:05; routed=incident
- `r12-sealed-063` category=combo; style=chatty; text=Boutique on Nov 22 at 14:07, could you give me five suspects and leave out adservice?; gold={'at': '--11-22T14:07', 'exclude_services': ['adservice'], 'system': 'online-boutique', 'top_k': 5, 'want_fault_type': True}; pred={'system': 'online-boutique', 'at': '2024-11-22T14:07', 'top_k': 5, 'exclude_services': ['adservice'], 'want_fault_type': True}; status=ok; routed=incident

decision (1 shown):

- `r12-sealed-106` category=refuse; style=terse; text=boutique trouble reported for 25 Jan, 10:00 UTC - real?; status=clarify; reason=Do you have a ticket number for this incident?; routed=incident

round_trip (11 shown):

- `r12-sealed-019` summary=Investigation of the train-ticket incident, requested by time 2024-12-08T09:15 (ticket INC-2047), top 3 suspects requested. The suspected services, most likely first, are ts-admin-travel-service, ts-order-service, ts-tr...; recon_status=ok; recon_params={'system': 'train-ticket', 'ticket': 'INC-2047', 'at': '2024-12-08T09:15', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; gold={'at': '2024-12-08T09:15', 'exclude_services': [], 'system': 'train-ticket', 'top_k': 3, 'want_fault_type': True}
- `r12-sealed-020` summary=Investigation of the train-ticket incident, requested by time 2024-12-08T14:33 (ticket INC-2056), top 3 suspects requested. The suspected services, most likely first, are ts-route-service, ts-inside-payment-service, ts-...; recon_status=ok; recon_params={'system': 'train-ticket', 'ticket': 'INC-2056', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; gold={'at': '2024-12-08T14:33', 'exclude_services': [], 'system': 'train-ticket', 'top_k': 3, 'want_fault_type': True}
- `r12-sealed-021` summary=Investigation of the train-ticket incident, requested by time 2024-12-10T12:03 (ticket INC-2060), top 3 suspects requested. The suspected services, most likely first, are ts-order-other-service, ts-auth-service, ts-orde...; recon_status=ok; recon_params={'system': 'train-ticket', 'ticket': 'INC-2060', 'at': '2024-12-10T12:03', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; gold={'at': '2024-12-10T12:03', 'exclude_services': [], 'system': 'train-ticket', 'top_k': 3, 'want_fault_type': True}
- `r12-sealed-022` summary=Investigation of the train-ticket incident, requested by time 2024-12-08T07:30 (ticket INC-2044), top 3 suspects requested. The suspected services, most likely first, are ts-auth-service, ts-order-service, ts-admin-basi...; recon_status=ok; recon_params={'system': 'train-ticket', 'ticket': 'INC-2044', 'at': '2024-12-08T07:30', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; gold={'at': '2024-12-08T07:30', 'exclude_services': [], 'system': 'train-ticket', 'top_k': 3, 'want_fault_type': True}
- `r12-sealed-039` summary=Investigation of the train-ticket incident, requested by time 2024-12-10T12:38 (ticket INC-2061), top 3 suspects requested. The suspected services, most likely first, are ts-order-service, ts-auth-service, ts-travel2-se...; recon_status=ok; recon_params={'system': 'train-ticket', 'ticket': 'INC-2061', 'at': '2024-12-10T12:38', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; gold={'at': '2024-12-10T12:38', 'exclude_services': [], 'system': 'train-ticket', 'top_k': 3, 'want_fault_type': True}
- `r12-sealed-040` summary=Investigation of the train-ticket incident, requested by time 2024-12-10T13:14 (ticket INC-2062), top 3 suspects requested. The suspected services, most likely first, are ts-admin-basic-info-service, ts-order-service, t...; recon_status=ok; recon_params={'system': 'train-ticket', 'ticket': 'INC-2062', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; gold={'at': '2024-12-10T13:14', 'exclude_services': [], 'system': 'train-ticket', 'top_k': 3, 'want_fault_type': True}
- `r12-sealed-042` summary=Investigation of the train-ticket incident, requested by time 2024-12-10T12:45 (ticket INC-2061), top 3 suspects requested. The suspected services, most likely first, are ts-order-service, ts-auth-service, ts-travel2-se...; recon_status=ok; recon_params={'system': 'train-ticket', 'ticket': 'INC-2061', 'at': '2024-12-10T12:45', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; gold={'at': '2024-12-10T12:45', 'exclude_services': [], 'system': 'train-ticket', 'top_k': 3, 'want_fault_type': True}
- `r12-sealed-043` summary=Investigation of the train-ticket incident, requested by time 2024-12-10T12:03 (ticket INC-2060), top 3 suspects requested. The suspected services, most likely first, are ts-order-other-service, ts-auth-service, ts-orde...; recon_status=ok; recon_params={'system': 'train-ticket', 'ticket': 'INC-2060', 'at': '2024-12-10T12:03', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; gold={'at': '2024-12-10T12:03', 'exclude_services': [], 'system': 'train-ticket', 'top_k': 3, 'want_fault_type': True}
- `r12-sealed-044` summary=Investigation of the train-ticket incident, requested by time 2024-12-10T13:10 (ticket INC-2062), top 3 suspects requested. The suspected services, most likely first, are ts-admin-basic-info-service, ts-order-service, t...; recon_status=ok; recon_params={'system': 'train-ticket', 'ticket': 'INC-2062', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; gold={'at': '2024-12-10T13:10', 'exclude_services': [], 'system': 'train-ticket', 'top_k': 3, 'want_fault_type': True}
- `r12-sealed-057` summary=Investigation of the train-ticket incident, requested by time 2024-12-08T12:47 (ticket INC-2053), top 3 suspects requested. The suspected services, most likely first, are ts-route-service, ts-inside-payment-service, ts-...; recon_status=ok; recon_params={'system': 'train-ticket', 'ticket': 'INC-2053', 'top_k': 3, 'exclude_services': ['ts-seat-service'], 'want_fault_type': True}; gold={'at': '2024-12-08T12:47', 'exclude_services': ['ts-seat-service'], 'system': 'train-ticket', 'top_k': 3, 'want_fault_type': True}
- `r12-sealed-062` summary=Investigation of the train-ticket incident, requested by time 2024-12-10T12:38 (ticket INC-2061), top 2 suspects requested. The suspected services, most likely first, are ts-auth-service, ts-travel2-service; no fault ty...; recon_status=ok; recon_params={'system': 'train-ticket', 'ticket': 'INC-2061', 'top_k': 2, 'want_fault_type': False, 'exclude_services': ['ts-order-service']}; gold={'at': '2024-12-10T12:38', 'exclude_services': ['ts-order-service'], 'system': 'train-ticket', 'top_k': 2, 'want_fault_type': False}

facts (3 shown):

- `r12-sealed-068` summary=Investigation of the sock-shop incident, requested by ticket INC-1112, top 3 suspects requested. The suspected services, most likely first, are user, orders, front-end; fault type delay (probability 0.47); onset 23:10 U...; missing=['exclusions']
- `r12-sealed-070` summary=Investigation of the sock-shop incident, requested by ticket INC-1104, top 3 suspects requested. The suspected services, most likely first, are user, orders, payment; fault type cpu (probability 0.39); onset 19:23 UTC. ...; missing=['exclusions']
- `r12-sealed-072` summary=Investigation of the sock-shop incident, requested by time 2024-01-20T14:21 (ticket INC-1094), top 3 suspects requested. The suspected services, most likely first, are payment, carts-db, user; fault type disk (probabili...; missing=['exclusions']

### keyword/regex parser

**KILL** (RE3 exact match 83.3% < 85%; RE3 NL task score significantly below the gold run)

| Criterion | Target | Measured | Result |
|---|---|---|---|
| exact match, RE3 (new system and incidents) | >= 90% (kill < 85%) | 83.3% (n=66, 95% CI 72.6%-90.4%) | not met |
| exact match, RE2 (regression) | >= 90% (kill < 85%) | 92.0% (n=25, 95% CI 75.0%-97.8%) | PASS |
| domain routing | >= 98% | 98.0% (n=252, 95% CI 95.4%-99.1%) | PASS |
| task score (root top-1), RE3, NL vs gold | lower CI bound of NL - gold > -5 points | NL 66.7% vs gold 74.2%; diff -7.6 [-13.6, -1.5] points (n=66) | not met |
| task score (joint), RE2, NL vs gold | lower CI bound of NL - gold > -5 points | NL 60.0% vs gold 64.0%; diff -4.0 [-12.0, +0.0] points (n=25) | not met |
| round trip | >= 90% | n/a | n/a |
| summary field fidelity | >= 90% of facts | n/a | n/a |
| warm end-to-end p95 / peak VRAM (reported) | - | n/a | - |

Routing by desk: incident 100.0% (97/97); plant 98.7% (75/76); squad 94.9% (75/79).

Exact match by system: online-boutique (re2) 100.0% (13/13); online-boutique (re3) 87.5% (14/16); sock-shop (re2) 83.3% (10/12); sock-shop (re3) 83.3% (15/18); train-ticket (re3) 81.2% (26/32).

Final decision on all 107 items: 87.9% (94/107). clarify: 100.0% (6/6) {'clarify': 6}. refuse: 60.0% (6/10) {'clarify': 2, 'ok': 2, 'refused': 6}. spec: 90.1% (82/91) {'clarify': 8, 'ok': 82, 'refused': 1}.

| Category | Exact match, all spec items | Decision (all kinds) |
|---|---|---|
| clarify | - | 100.0% (6/6) |
| combo | 100.0% (6/6) | 100.0% (6/6) |
| exclude | 90.9% (10/11) | 100.0% (11/11) |
| fault | 80.0% (4/5) | 100.0% (5/5) |
| partial | 81.8% (18/22) | 81.8% (18/22) |
| refuse | - | 60.0% (6/10) |
| relative | 66.7% (4/6) | 66.7% (4/6) |
| ticket | 90.5% (19/21) | 90.5% (19/21) |
| time | 91.7% (11/12) | 91.7% (11/12) |
| topk | 75.0% (6/8) | 100.0% (8/8) |

#### keyword/regex parser: failure examples

translation (12 shown):

- `r12-sealed-005` category=ticket; style=words; text=Ticket two thousand and twenty-four, the boutique.; gold={'exclude_services': [], 'system': 'online-boutique', 'ticket': 'INC-2024', 'top_k': 3, 'want_fault_type': True}; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'online-boutique'}; status=clarify; reason=Which incident? Give a ticket (INC-nnnn) or a date and time.; routed=incident
- `r12-sealed-022` category=time; style=chatty; text=Look at the Train Ticket incident from December 8th 2024, around half past seven in the morning.; gold={'at': '2024-12-08T07:30', 'exclude_services': [], 'system': 'train-ticket', 'top_k': 3, 'want_fault_type': True}; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'at': '2024-12-08', 'system': 'train-ticket'}; status=clarify; reason=18 train-ticket incidents on 2024-12-08: INC-2042 (2024-12-08T06:18), INC-2043 (2024-12-08T06:53), INC-2044 (2024-12-08T07:28), INC-2045 (2024-12-08T08:04) and 14 more. Which one?; routed=incident
- `r12-sealed-024` category=partial; style=chatty; text=The boutique, November 23rd around 3:13 in the afternoon - can you dig into it?; gold={'at': '--11-23T15:13', 'exclude_services': [], 'system': 'online-boutique', 'top_k': 3, 'want_fault_type': True}; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'online-boutique', 'at': '--11-23T03:13'}; status=refused; reason=no online-boutique incident matching --11-23T03:13 at or before 2024-12-11T09:00; routed=incident
- `r12-sealed-025` category=partial; style=terse; text=sock shop incident, nov 22nd, 7 am sharp; gold={'at': '--11-22T07:00', 'exclude_services': [], 'system': 'sock-shop', 'top_k': 3, 'want_fault_type': True}; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'sock-shop', 'at': '--11-22'}; status=clarify; reason=5 sock-shop incidents matching --11-22 at or before 2024-12-11T09:00: INC-2001 (2024-11-22T05:42), INC-2002 (2024-11-22T06:08), INC-2003 (2024-11-22T06:34), INC-2004 (2024-11-22T07:00) and 1 more. Which one?; routed=incident
- `r12-sealed-029` category=partial; style=words; text=Ten December, train ticket system, twelve thirty-eight.; gold={'at': '--12-10T12:38', 'exclude_services': [], 'system': 'train-ticket', 'top_k': 3, 'want_fault_type': True}; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'train-ticket'}; status=clarify; reason=Which incident? Give a ticket (INC-nnnn) or a date and time.; routed=incident
- `r12-sealed-038` category=partial; style=keyvalue; text=system=train-ticket at=---08T15:44; gold={'at': '---08T15:44', 'exclude_services': [], 'system': 'train-ticket', 'top_k': 3, 'want_fault_type': True}; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'train-ticket'}; status=clarify; reason=Which incident? Give a ticket (INC-nnnn) or a date and time.; routed=incident
- `r12-sealed-042` category=relative; style=chatty; text=Hi, the train ticket booking thing keeps coming up - the one from yesterday at about a quarter to one in the afternoon?; gold={'at': '2024-12-10T12:45', 'exclude_services': [], 'system': 'train-ticket', 'top_k': 3, 'want_fault_type': True}; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'at': '2024-12-10', 'system': 'train-ticket'}; status=clarify; reason=3 train-ticket incidents on 2024-12-10: INC-2060 (2024-12-10T12:03), INC-2061 (2024-12-10T12:38), INC-2062 (2024-12-10T13:14). Which one?; routed=incident
- `r12-sealed-043` category=relative; style=words; text=Train ticket, yesterday at twelve oh three in the afternoon.; gold={'at': '2024-12-10T12:03', 'exclude_services': [], 'system': 'train-ticket', 'top_k': 3, 'want_fault_type': True}; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'at': '2024-12-10', 'system': 'train-ticket'}; status=clarify; reason=3 train-ticket incidents on 2024-12-10: INC-2060 (2024-12-10T12:03), INC-2061 (2024-12-10T12:38), INC-2062 (2024-12-10T13:14). Which one?; routed=incident
- `r12-sealed-045` category=topk; style=plain; text=Give me just the 2 likeliest services for sock shop ticket INC-2018.; gold={'exclude_services': [], 'system': 'sock-shop', 'ticket': 'INC-2018', 'top_k': 2, 'want_fault_type': True}; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'sock-shop', 'ticket': 'INC-2018'}; status=ok; routed=incident
- `r12-sealed-048` category=topk; style=question; text=Who is the single most likely culprit in INC-2048, train ticket?; gold={'exclude_services': [], 'system': 'train-ticket', 'ticket': 'INC-2048', 'top_k': 1, 'want_fault_type': True}; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'ticket': 'INC-2048', 'system': 'train-ticket'}; status=ok; routed=incident
- `r12-sealed-061` category=fault; style=terse; text=sock shop INC-2040 - don't bother with the fault kind; gold={'exclude_services': [], 'system': 'sock-shop', 'ticket': 'INC-2040', 'top_k': 3, 'want_fault_type': False}; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'sock-shop', 'ticket': 'INC-2040'}; status=ok; routed=incident
- `r12-sealed-070` category=ticket; style=words; text=Ticket one thousand one hundred and four, sock shop.; gold={'exclude_services': [], 'system': 'sock-shop', 'ticket': 'INC-1104', 'top_k': 3, 'want_fault_type': True}; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'sock-shop'}; status=clarify; reason=Which incident? Give a ticket (INC-nnnn) or a date and time.; routed=incident

decision (4 shown):

- `r12-sealed-101` category=refuse; style=plain; text=Please restart ts-auth-service on the train ticket system.; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'train-ticket'}; status=clarify; reason=Which incident? Give a ticket (INC-nnnn) or a date and time.; routed=incident
- `r12-sealed-102` category=refuse; style=formal; text=I need nine suspects for INC-2050, train ticket.; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'ticket': 'INC-2050', 'system': 'train-ticket'}; status=ok; routed=incident
- `r12-sealed-103` category=refuse; style=plain; text=Investigate train ticket INC-2051 and exclude ts-flux-capacitor.; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'ticket': 'INC-2051', 'system': 'train-ticket'}; status=ok; routed=incident
- `r12-sealed-104` category=refuse; style=chatty; text=Look at the sock shop and the train ticket system together for the 8th of December.; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'train-ticket', 'at': '--12-08'}; status=clarify; reason=18 train-ticket incidents matching --12-08 at or before 2024-12-11T09:00: INC-2042 (2024-12-08T06:18), INC-2043 (2024-12-08T06:53), INC-2044 (2024-12-08T07:28), INC-2045 (2024-12-08T08:04) and 14 more. Which one?; routed=incident

## Supplementary: qwen3-coder:30b as translator (outside the verdict)

Queued after the G10 verdict (job `r12-q30-sealed`, pinned 56e60c72, same code and prompts as the 14B run). It ran once on the sealed set and is not part of G10, which names the 14B.

| Translator | Exact RE3 | Exact RE2 | Routing | NL − gold, RE3 | NL − gold, RE2 | Decisions (107) | Round trip | Fidelity | Warm p95 |
|---|---|---|---|---|---|---|---|---|---|
| qwen3-coder:30b | 93.9% (62/66) | 92.0% (23/25) | 100.0% (252/252) | −3.0 [−7.6, +0.0] | +0.0 [+0.0, +0.0] | 98.1% (105/107) | 95.5% (85/89) | 100.0% | 6.96 s |

- **It passes both exact-match bars** that the 14B missed. Train Ticket is 93.8% (30/32), and none of the 30B's misses is an added year.
- **Its task-score miss on RE3** comes from two terse requests, "shop inc 2033" and "shop 23rd 21:35". With a third system on the desk it asked which system "shop" meant instead of reading it as the sock shop.
- **Results:** every refusal item was refused (10/10).
- **The 14B's G8b gap is the same here:** a larger translator stops writing values the user did not say.

## Analysis

<!-- analysis:start -->
### G9

**G9 is a kill, by the pre-registered rule.** The newcomer helps the new incidents a lot and the old ones a little less, but the RE2 loss is significant.

| Test split | Joined − old team | Fixed | Broken |
|---|---|---|---|
| RE3 code-level faults | +22.6 [+12.9, +33.9] (56.5 → 79.0) | 14 | 0 |
| RE2 resource faults | −3.3 [−6.7, −0.8] top-1 and joint | 0 | 4 |

- **The RE3 gain passes** (criterion 1).
- **The RE2 loss fails** both non-inferiority criteria (lower bound −6.7 < −5). Its CI also lies wholly below 0, which is the kill.
- **Pooled:** over all 182 test incidents, top-1 goes from 77.5% to 83.0%.
- **The old team is untouched.** Its state hash is identical before and after the newcomer joined and ran. The whole RE2 loss comes from the joined centre's rule.

**Why the newcomer costs anything on RE2: packet loss.**
- On RE2 the gate let the log part in for 18 of 120 incidents, and 17 of those were network packet-loss faults.
- All 4 broken incidents are loss faults:

  | Incident | Joined top-1 |
  |---|---|
  | OB checkoutservice | frontend |
  | OB emailservice | checkoutservice |
  | SS payment | orders |
  | SS user | front-end |

- **Why the gate opens on loss:** loss leaves weak metric anomalies (max z below the gate's 45), so the gate trusts the logs.
- **Why the logs then mislead:** the new error lines appear in the services that *call* the faulty one. The newcomer counts new error templates per service, so it points at a caller, one hop upstream.
- **The same mechanism was the untested hypothesis for Train Ticket in G4e.** Here it is observed directly.
- **On RE3 the log part leads to the right service.** Code-level faults log new errors in the faulty service itself. There the log part took part in 85.5% of incidents and changed the top-1 in 27.4%, every time for the better.

**The rule gate cannot tell a code fault from a packet-loss fault.** It sees only the reports:
- both incident classes can have weak metrics;
- both have new error templates;
- what differs is *where* the errors are logged relative to the fault. A count of new templates per service does not carry that.

The dev split showed the same trade-off before the test. No candidate rule gained on RE3 without losing on RE2: the chosen rule lost 5.0 points on RE2 dev, and the one rule with no RE2 loss (`rr_w0.5`) gained only 3.6 on RE3 dev. The test confirmed what dev predicted.

**Two smaller points:**
- **The joined team equals the RE3-only fusion on RE3:** 79.0 against 77.4, +1.6 [−6.5, +9.7]. So the gate costs RE3 nothing.
- **The RE3-only fusion scores 77.4 here, against 80.6 in G4d.** It is recomputed here from the team's object-layer reports; the 3-point (two-incident) difference was not investigated.

**What G9 says about growth.** A specialist written in code joins the running team through the command centre with no retraining and no change to the old members. It brings a large gain on the incident class it was built for. But joining is not free when the newcomer's signal also fires, misleadingly, on the old incident class, and a centre that sees only counts cannot separate the two cases.

The fix the evidence points to is a richer *report*, not a cleverer gate. The newcomer should say which service its new errors are *about*, since a caller's error lines usually name the callee, not only which service logged them. Testing that needs incidents not used here: the RE2 Train Ticket incidents, whose roots were never ranked in a test, could serve as the fresh RE2-side check.

### G10

**G10 is not passed and not killed.**

| Criterion (qwen2.5-coder:14b) | Pass bar | Result |
|---|---|---|
| Exact match, RE3 items | ≥ 90% (kill < 85%) | 87.9% [77.9, 93.7] |
| Exact match, RE2 items | ≥ 90% (kill < 85%) | 88.0% [70.0, 95.8] |
| RE2 task score, NL − gold (25 items) | lower bound > −5 | −4.0 [−12.0, +0.0] |
| Routing | ≥ 98% | 100% (252/252) |
| RE3 task score, NL vs gold | lower bound > −5 | equal (74.2 vs 74.2 root top-1) |
| Round trip | ≥ 90% | 95.6% |
| Fidelity | ≥ 90% | 100% of 669 facts |

- Exact match on both groups sits between the pass and kill lines.
- The RE2 task-score miss comes from one refused request among 25 items.
- Also right: 106 of 107 decisions; p95 is 4.4 s.

**The desk did grow without prompt edits.** Train Ticket was added as one entry in the systems table.
- The 14B routed every request correctly.
- It extracted 29 of 32 Train Ticket specs exactly (90.6%), the best of the three systems.
- Every request about a new incident that ran named the right incident.

Growth needed one thing first, found in review before any LLM saw the R12 sets. Every place a prompt names a system (the aliases and services paragraph, the refusal rule, the spec description, the router line) had to be rendered from data. The v2 prompts named the two systems in prose, and under the no-edit rule that prose would have told the model only two systems exist.

**One habit keeps exact match below 90%: the model adds a year.** Of the 14B's 11 exact misses:
- **8 are month-and-day dates written with 2024 added** ("Nov 22 at 11:59" became `2024-11-22T11:59` instead of `--11-22T11:59`). In G8b this happened on 2 of 20 time items.
  - It may be more frequent here because the desk clock and the new incidents are in the same months; that is not tested.
  - Every one resolves to the same incident, because the catalogue holds one year of incidents. The task score is unaffected.
- **"February 20th, 23:49" lost its month** (`---20T23:49`). The time still picks the right incident.
- **January was added to "the 19th" twice.** One of these (sealed-089) also put "no fault type" into the exclusion list, the same misreading as G8b's sealed-116. The desk refused it, because no service has that name.

That one refusal is the whole RE2 task-score gap: 1 of 25 items, which is why the CI reaches −12.

**gemma4:e4b and the keyword parser fail G10.**

| Translator | RE3 exact | RE3 task score vs gold |
|---|---|---|
| gemma4:e4b | 81.8% | −9.1 |
| keyword parser | 83.3% | −7.6 |

As in G8b, gemma refuses or asks for a ticket on dates it cannot format.

**What G10 says.**
- **Growing the desk works when the prompt's domain facts are data:** no routing loss, and no penalty on the new system.
- **The pass line is decided by an old error class, not by growth:** the translator writes a year the user did not say.
- **Two fixes, both outside this run:**
  - Scale: qwen3-coder:30b made no date errors on the G8b set, but it is too slow on this card.
  - A literal check in the desk, in code: an extracted value that the request text does not contain (here a year) is dropped back to the partial form or questioned. Being a code change, it would need a fresh sealed set to be judged.
<!-- analysis:end -->
