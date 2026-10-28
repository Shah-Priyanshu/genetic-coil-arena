# R11 v2: the ecosystem on real incidents, partial dates resolved by the desk (goal G8b)

Generated 2026-10-27 21:03. Command: `python scripts/r11_ecosystem.py report --spec-version 2`.

One shared translator picks the domain (squad, plant or incident), extracts the GoalSpec, the command centre runs `rca.investigate` through the G4c trained team on a real RCAEval RE2 incident, the reverse translator summarises the trace, and the summary is routed and translated again.

- **Sealed set:** 142 hand-written incident requests (`experiments/data/r11b_sealed_test.jsonl`, SHA-256 `78e48e2755bd6718683d64d61c262edc1c3c0c99dc7d06dce40fc82a7af8639c`), plus the R7a sealed spec and clarify items for the routing check (SHA-256 `ca6ff07b5fb66ef21c288ba9a82d5e7763992fe2142b1a582baf597e542b5e01`). Each translator ran once.
- **Catalogue:** 120 test incidents (SS/OB repetitions 2 and 3), fingerprint `409929e01f894a7e24e60398f7825551a502f3e7a6dde1a1f121df1236c60edf`.
- **Team:** the G4c trained metric fault specialist (config {'hidden': 0, 'l2': 1.0, 'epochs': 400}, trained on 150 Train Ticket and SS/OB repetition-1 incidents; leave-one-system-out accuracy OB 70.0%, SS 66.7%, TT 56.7%); through the object layer it gets top-1 88.3% and joint 68.3% on all 120 catalogued incidents (R10 step 3 reported 68.3 joint).

## Verdict

| Translator | G8b verdict | Exact match | Routing | NL - gold task score | Round trip | Fidelity | Warm p95 |
|---|---|---|---|---|---|---|---|
| qwen2.5-coder:14b (refusal policy) | **PASS** | 92.5% (111/120) | 100.0% (283/283) | +0.0 [+0.0, +0.0] | 98.3% (116/118) | 100.0% | 4.76 s |
| gemma4:e4b (refusal policy) | **KILL** | 83.3% (100/120) | 99.6% (282/283) | -2.5 [-5.8, +0.0] | 98.2% (109/111) | 99.3% | 2.70 s |
| keyword/regex parser | **KILL** | 82.5% (99/120) | 97.9% (277/283) | -4.2 [-8.3, -0.8] | n/a | n/a | n/a |

## Per translator

### qwen2.5-coder:14b (refusal policy)

**PASS**

| Criterion | Target | Measured | Result |
|---|---|---|---|
| exact match | >= 90% (kill < 85%) | 92.5% (n=120, 95% CI 86.4%-96.0%) | PASS |
| domain routing | >= 98% | 100.0% (n=283, 95% CI 98.7%-100.0%) | PASS |
| task score, NL vs gold | lower CI bound of NL - gold > -5 points | NL 63.3% vs gold 63.3%; diff +0.0 [+0.0, +0.0] points (n=120) | PASS |
| round trip | >= 90% | 98.3% (n=118, 95% CI 94.0%-99.5%) | PASS |
| summary field fidelity | >= 90% of facts (kill < 75%) | 100.0% of 881 facts; all facts in 100.0% of 118 summaries | PASS |
| warm end-to-end p95 | <= 10 s | 4.76 s (p50 4.38 s: translate 2.22, team 0.04, summary 2.13) | PASS |
| peak VRAM | <= 12 GB | 9.8 GB | PASS |

Routing by desk: incident 100.0% (128/128); plant 100.0% (76/76); squad 100.0% (79/79).

Final decision on all 142 items (spec answered with a spec, clarify with a question, refuse with a refusal): 97.2% (138/142). clarify: 100.0% (8/8) {'clarify': 8}. refuse: 85.7% (12/14) {'clarify': 1, 'ok': 1, 'refused': 12}. spec: 98.3% (118/120) {'clarify': 1, 'ok': 118, 'refused': 1}.

Task score: NL joint accuracy 63.3% (top-1 83.3%), gold 63.3% (top-1 84.2%) over 120 requests; 2 had no run (refused, clarified or invalid) and 0 named a different incident than the gold.

| Category | Exact match | Decision (all kinds) |
|---|---|---|
| clarify | - | 100.0% (8/8) |
| combo | 80.0% (8/10) | 90.0% (9/10) |
| exclude | 100.0% (12/12) | 100.0% (12/12) |
| fault | 100.0% (8/8) | 100.0% (8/8) |
| partial | 87.5% (21/24) | 95.8% (23/24) |
| refuse | - | 85.7% (12/14) |
| relative | 75.0% (6/8) | 100.0% (8/8) |
| ticket | 100.0% (24/24) | 100.0% (24/24) |
| time | 90.0% (18/20) | 100.0% (20/20) |
| topk | 100.0% (14/14) | 100.0% (14/14) |

| Style | Exact match |
|---|---|
| chatty | 94.1% (16/17) |
| formal | 77.8% (7/9) |
| keyvalue | 100.0% (3/3) |
| plain | 95.1% (39/41) |
| question | 100.0% (8/8) |
| terse | 87.1% (27/31) |
| typo | 100.0% (5/5) |
| words | 100.0% (6/6) |

Field accuracy on spec items: at 88.1%, exclude_services 98.3%, system 98.3%, ticket 98.2%, top_k 98.3%, want_fault_type 98.3%.

Fact accuracy: at 100.0%, exclusions 100.0%, fault_probability 100.0%, fault_type 100.0%, onset 100.0%, suspects 100.0%, system 100.0%, ticket 100.0%.

#### qwen2.5-coder:14b (refusal policy): failure examples

translation (9 shown):

- `r11b-sealed-028` category=time; style=formal; text=Please investigate the Sock Shop incident of 20 January at 12:55 hours.; gold={'at': '--01-20T12:55', 'exclude_services': [], 'system': 'sock-shop', 'top_k': 3, 'want_fault_type': True}; pred={'system': 'sock-shop', 'at': '2024-01-20T12:55', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; status=ok; routed=incident
- `r11b-sealed-032` category=time; style=plain; text=Check the sock shop failure on 21 January at 07:51.; gold={'at': '--01-21T07:51', 'exclude_services': [], 'system': 'sock-shop', 'top_k': 3, 'want_fault_type': True}; pred={'system': 'sock-shop', 'at': '2024-01-21T07:51', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; status=ok; routed=incident
- `r11b-sealed-054` category=partial; style=terse; text=the 19th, 12:55, online boutique; gold={'at': '---19T12:55', 'exclude_services': [], 'system': 'online-boutique', 'top_k': 3, 'want_fault_type': True}; pred={'system': 'online-boutique', 'at': '--01-19T12:55', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; status=ok; routed=incident
- `r11b-sealed-059` category=partial; style=formal; text=Regarding the Online Boutique, the incident on the 19th at 20:07, kindly diagnose.; gold={'at': '---19T20:07', 'exclude_services': [], 'system': 'online-boutique', 'top_k': 3, 'want_fault_type': True}; pred={'system': 'online-boutique', 'at': '--01-19T20:07', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; status=ok; routed=incident
- `r11b-sealed-066` category=partial; style=terse; text=boutique, February 20th; gold={'at': '--02-20', 'exclude_services': [], 'system': 'online-boutique', 'top_k': 3, 'want_fault_type': True}; pred={'system': 'online-boutique', 'at': '---20', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; status=clarify; reason=---20 matches online-boutique incidents on 2 dates (2024-01-20, 2024-02-20). Which date do you mean?; routed=incident
- `r11b-sealed-070` category=relative; style=terse; text=Two days ago, 23:49, boutique incident - root cause?; gold={'at': '2024-02-20T23:49', 'exclude_services': [], 'system': 'online-boutique', 'top_k': 3, 'want_fault_type': True}; pred={'system': 'online-boutique', 'at': '---20T23:49', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; status=ok; routed=incident
- `r11b-sealed-075` category=relative; style=terse; text=boutique, 2 days ago, 11:50 pm; gold={'at': '2024-02-20T23:50', 'exclude_services': [], 'system': 'online-boutique', 'top_k': 3, 'want_fault_type': True}; pred={'system': 'online-boutique', 'at': '---20T23:50', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; status=ok; routed=incident
- `r11b-sealed-112` category=combo; style=plain; text=Online boutique INC-1030 (it was on Jan 19 around 10:30): top 4, leave out redis.; gold={'at': '--01-19T10:30', 'exclude_services': ['redis'], 'system': 'online-boutique', 'ticket': 'INC-1030', 'top_k': 4, 'want_fault_type': True}; pred={'system': 'online-boutique', 'ticket': 'INC-1030', 'top_k': 4, 'exclude_services': ['redis'], 'want_fault_type': True}; status=ok; routed=incident
- `r11b-sealed-116` category=combo; style=chatty; text=Right, INC-1016 on the online boutique - three suspects but not the redis, and I do not want the fault type.; gold={'exclude_services': ['redis'], 'system': 'online-boutique', 'ticket': 'INC-1016', 'top_k': 3, 'want_fault_type': False}; pred={'system': 'online-boutique', 'ticket': 'INC-1016', 'top_k': 3, 'exclude_services': ['redis', 'want_fault_type'], 'want_fault_type': True}; status=refused; reason=online-boutique has no service named want_fault_type; routed=incident

decision (2 shown):

- `r11b-sealed-130` category=refuse; style=plain; text=Online boutique INC-1033 with the top 8 suspects.; pred={'system': 'online-boutique', 'ticket': 'INC-1033', 'top_k': 5, 'want_fault_type': True, 'exclude_services': []}; status=ok; routed=incident
- `r11b-sealed-139` category=refuse; style=plain; text=Investigate the boutique incident tomorrow at 10:00.; status=clarify; reason=Which incident? Give a ticket or a date and time.; routed=incident

round_trip (2 shown):

- `r11b-sealed-097` summary=Investigation of the online-boutique incident, requested by ticket INC-1024, top 3 suspects requested. The suspected services, most likely first, are emailservice, checkoutservice, frontend; fault type mem (probability ...; recon_status=ok; recon_params={'system': 'online-boutique', 'ticket': 'INC-1024', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; gold={'exclude_services': ['adservice', 'currencyservice'], 'system': 'online-boutique', 'ticket': 'INC-1024', 'top_k': 3, 'want_fault_type': True}
- `r11b-sealed-111` summary=Investigation of the sock-shop incident, requested by time ---20T14:48 (ticket INC-1095), top 3 suspects requested. The suspected services, most likely first, are payment, carts, queue-master; fault type disk (probabili...; recon_status=ok; recon_params={'system': 'sock-shop', 'at': '---20T14:48', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; gold={'at': '---20T14:48', 'exclude_services': [], 'system': 'sock-shop', 'ticket': 'INC-1095', 'top_k': 3, 'want_fault_type': True}

### gemma4:e4b (refusal policy)

**KILL** (exact match 83.3% < 85%)

| Criterion | Target | Measured | Result |
|---|---|---|---|
| exact match | >= 90% (kill < 85%) | 83.3% (n=120, 95% CI 75.7%-88.9%) | not met |
| domain routing | >= 98% | 99.6% (n=283, 95% CI 98.0%-99.9%) | PASS |
| task score, NL vs gold | lower CI bound of NL - gold > -5 points | NL 60.8% vs gold 63.3%; diff -2.5 [-5.8, +0.0] points (n=120) | not met |
| round trip | >= 90% | 98.2% (n=111, 95% CI 93.7%-99.5%) | PASS |
| summary field fidelity | >= 90% of facts (kill < 75%) | 99.3% of 825 facts; all facts in 94.6% of 111 summaries | PASS |
| warm end-to-end p95 | <= 10 s | 2.70 s (p50 2.43 s: translate 1.27, team 0.03, summary 1.13) | PASS |
| peak VRAM | <= 12 GB | 5.1 GB | PASS |

Routing by desk: incident 100.0% (128/128); plant 98.7% (75/76); squad 100.0% (79/79).

Final decision on all 142 items (spec answered with a spec, clarify with a question, refuse with a refusal): 93.0% (132/142). clarify: 100.0% (8/8) {'clarify': 8}. refuse: 92.9% (13/14) {'clarify': 1, 'refused': 13}. spec: 92.5% (111/120) {'clarify': 4, 'ok': 111, 'refused': 5}.

Task score: NL joint accuracy 60.8% (top-1 78.3%), gold 63.3% (top-1 84.2%) over 120 requests; 9 had no run (refused, clarified or invalid) and 0 named a different incident than the gold.

| Category | Exact match | Decision (all kinds) |
|---|---|---|
| clarify | - | 100.0% (8/8) |
| combo | 90.0% (9/10) | 100.0% (10/10) |
| exclude | 75.0% (9/12) | 91.7% (11/12) |
| fault | 100.0% (8/8) | 100.0% (8/8) |
| partial | 70.8% (17/24) | 87.5% (21/24) |
| refuse | - | 92.9% (13/14) |
| relative | 87.5% (7/8) | 87.5% (7/8) |
| ticket | 100.0% (24/24) | 100.0% (24/24) |
| time | 65.0% (13/20) | 80.0% (16/20) |
| topk | 92.9% (13/14) | 100.0% (14/14) |

| Style | Exact match |
|---|---|
| chatty | 82.4% (14/17) |
| formal | 77.8% (7/9) |
| keyvalue | 33.3% (1/3) |
| plain | 92.7% (38/41) |
| question | 87.5% (7/8) |
| terse | 80.6% (25/31) |
| typo | 80.0% (4/5) |
| words | 66.7% (4/6) |

Field accuracy on spec items: at 71.6%, exclude_services 92.5%, system 92.5%, ticket 100.0%, top_k 92.5%, want_fault_type 91.7%.

Fact accuracy: at 100.0%, exclusions 94.6%, fault_probability 100.0%, fault_type 100.0%, onset 100.0%, suspects 100.0%, system 100.0%, ticket 100.0%.

#### gemma4:e4b (refusal policy): failure examples

translation (12 shown):

- `r11b-sealed-026` category=time; style=plain; text=Boutique incident on January 17 at 21:50.; gold={'at': '--01-17T21:50', 'exclude_services': [], 'system': 'online-boutique', 'top_k': 3, 'want_fault_type': True}; pred={'system': 'online-boutique', 'at': '2024-01-17T21:50', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; status=ok; routed=incident
- `r11b-sealed-028` category=time; style=formal; text=Please investigate the Sock Shop incident of 20 January at 12:55 hours.; gold={'at': '--01-20T12:55', 'exclude_services': [], 'system': 'sock-shop', 'top_k': 3, 'want_fault_type': True}; status=refused; reason=the date format provided is invalid for the 'at' parameter; routed=incident
- `r11b-sealed-029` category=time; style=words; text=Online boutique, January nineteenth, twenty-seven past seven in the evening.; gold={'at': '--01-19T19:27', 'exclude_services': [], 'system': 'online-boutique', 'top_k': 3, 'want_fault_type': True}; pred={'system': 'online-boutique', 'at': '2024-01-19T19:27', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; status=ok; routed=incident
- `r11b-sealed-030` category=time; style=keyvalue; text=system=online-boutique when=Jan 18 13:10; gold={'at': '--01-18T13:10', 'exclude_services': [], 'system': 'online-boutique', 'top_k': 3, 'want_fault_type': True}; status=clarify; reason=Could you provide a ticket number or a specific incident identifier for the investigation?; routed=incident
- `r11b-sealed-031` category=time; style=terse; text=jan 19, 16:55, sock shop; gold={'at': '--01-19T16:55', 'exclude_services': [], 'system': 'sock-shop', 'top_k': 3, 'want_fault_type': True}; pred={'system': 'sock-shop', 'at': '2024-01-19T16:55', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; status=ok; routed=incident
- `r11b-sealed-034` category=time; style=typo; text=sock shop outge 20 jan 10:36 plese; gold={'at': '--01-20T10:36', 'exclude_services': [], 'system': 'sock-shop', 'top_k': 3, 'want_fault_type': True}; status=refused; reason=the date format provided is ambiguous for the required schema; routed=incident
- `r11b-sealed-039` category=time; style=words; text=Sock shop, twenty past two in the afternoon on 20 January.; gold={'at': '--01-20T14:20', 'exclude_services': [], 'system': 'sock-shop', 'top_k': 3, 'want_fault_type': True}; status=refused; reason=the date format provided is ambiguous for the required schema; routed=incident
- `r11b-sealed-045` category=partial; style=terse; text=sock shop, the 19th, 15:31; gold={'at': '---19T15:31', 'exclude_services': [], 'system': 'sock-shop', 'top_k': 3, 'want_fault_type': True}; pred={'system': 'sock-shop', 'at': '--01-19T15:31', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; status=ok; routed=incident
- `r11b-sealed-050` category=partial; style=chatty; text=Online boutique, on the 19th around a quarter to seven in the evening.; gold={'at': '---19T18:45', 'exclude_services': [], 'system': 'online-boutique', 'top_k': 3, 'want_fault_type': True}; pred={'system': 'online-boutique', 'at': '--01-19T18:45', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; status=ok; routed=incident
- `r11b-sealed-054` category=partial; style=terse; text=the 19th, 12:55, online boutique; gold={'at': '---19T12:55', 'exclude_services': [], 'system': 'online-boutique', 'top_k': 3, 'want_fault_type': True}; pred={'system': 'online-boutique', 'at': '--01-19T12:55', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; status=ok; routed=incident
- `r11b-sealed-057` category=partial; style=chatty; text=Early on the 21st (07:51) the sock shop failed; please investigate.; gold={'at': '---21T07:51', 'exclude_services': [], 'system': 'sock-shop', 'top_k': 3, 'want_fault_type': True}; status=refused; reason=the date format provided for 'early on the 21st' is ambiguous for the required 'at' parameter format.; routed=incident
- `r11b-sealed-059` category=partial; style=formal; text=Regarding the Online Boutique, the incident on the 19th at 20:07, kindly diagnose.; gold={'at': '---19T20:07', 'exclude_services': [], 'system': 'online-boutique', 'top_k': 3, 'want_fault_type': True}; pred={'system': 'online-boutique', 'at': '--01-19T20:07', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; status=ok; routed=incident

decision (1 shown):

- `r11b-sealed-139` category=refuse; style=plain; text=Investigate the boutique incident tomorrow at 10:00.; status=clarify; reason=What is the date for 'tomorrow'?; routed=incident

round_trip (2 shown):

- `r11b-sealed-111` summary=Investigation of the sock-shop incident, requested by time ---20T14:48 (ticket INC-1095), top 3 suspects requested. The suspected services, most likely first, are payment, carts, queue-master; fault type disk (probabili...; recon_status=ok; recon_params={'system': 'sock-shop', 'at': '---20T14:48', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; gold={'at': '---20T14:48', 'exclude_services': [], 'system': 'sock-shop', 'ticket': 'INC-1095', 'top_k': 3, 'want_fault_type': True}
- `r11b-sealed-112` summary=Investigation of the online-boutique incident, requested by time --01-19T10:30 (ticket INC-1030), top 4 suspects requested. The suspected services, most likely first, are checkoutservice, paymentservice, currencyservice...; recon_status=ok; recon_params={'system': 'online-boutique', 'at': '--01-19T10:30', 'top_k': 4, 'want_fault_type': True, 'exclude_services': ['redis']}; gold={'at': '--01-19T10:30', 'exclude_services': ['redis'], 'system': 'online-boutique', 'ticket': 'INC-1030', 'top_k': 4, 'want_fault_type': True}

facts (6 shown):

- `r11b-sealed-011` summary=Investigation of the sock-shop incident, requested by ticket INC-1109, top 3 suspects requested. The suspected services, most likely first, are user, user-db, shipping; fault type disk (probability 0.61); onset 21:44 UT...; missing=['exclusions']
- `r11b-sealed-019` summary=Investigation of the sock-shop incident, requested by ticket INC-1104, top 3 suspects requested. The suspected services, most likely first, are user, orders, payment; fault type cpu (probability 0.39); onset 19:23 UTC. ...; missing=['exclusions']
- `r11b-sealed-043` summary=Investigation of the sock-shop incident, requested by time --01-20T22:10 (ticket INC-1110), top 3 suspects requested. The suspected services, most likely first, are user, orders, front-end; fault type cpu (probability 0...; missing=['exclusions']
- `r11b-sealed-051` summary=Investigation of the sock-shop incident, requested by time ---20T23:37 (ticket INC-1113), top 3 suspects requested. The suspected services, most likely first, are user, orders, front-end; fault type delay (probability 0...; missing=['exclusions']
- `r11b-sealed-055` summary=Investigation of the sock-shop incident, requested by time ---20T19:23 (ticket INC-1104), top 3 suspects requested. The suspected services, most likely first, are user, orders, payment; fault type cpu (probability 0.39)...; missing=['exclusions']
- `r11b-sealed-079` summary=Investigation of the sock-shop incident, requested by time ---20T21:43 (ticket INC-1109), top 4 suspects requested. The suspected services, most likely first, are user, user-db, shipping, carts-db; fault type disk (prob...; missing=['exclusions']

### keyword/regex parser

**KILL** (exact match 82.5% < 85%; NL task score significantly below the gold run)

| Criterion | Target | Measured | Result |
|---|---|---|---|
| exact match | >= 90% (kill < 85%) | 82.5% (n=120, 95% CI 74.7%-88.3%) | not met |
| domain routing | >= 98% | 97.9% (n=283, 95% CI 95.5%-99.0%) | not met |
| task score, NL vs gold | lower CI bound of NL - gold > -5 points | NL 59.2% vs gold 63.3%; diff -4.2 [-8.3, -0.8] points (n=120) | not met |
| round trip | >= 90% | n/a | n/a |
| summary field fidelity | >= 90% of facts (kill < 75%) | n/a | n/a |
| warm end-to-end p95 | <= 10 s | n/a | n/a |
| peak VRAM | <= 12 GB | n/a | n/a |

Routing by desk: incident 99.2% (127/128); plant 98.7% (75/76); squad 94.9% (75/79).

Final decision on all 142 items (spec answered with a spec, clarify with a question, refuse with a refusal): 90.8% (129/142). clarify: 75.0% (6/8) {'clarify': 6, 'refused': 2}. refuse: 71.4% (10/14) {'clarify': 2, 'ok': 2, 'refused': 10}. spec: 94.2% (113/120) {'clarify': 6, 'ok': 113, 'refused': 1}.

Task score: NL joint accuracy 59.2% (top-1 79.2%), gold 63.3% (top-1 84.2%) over 120 requests; 7 had no run (refused, clarified or invalid) and 0 named a different incident than the gold.

| Category | Exact match | Decision (all kinds) |
|---|---|---|
| clarify | - | 75.0% (6/8) |
| combo | 80.0% (8/10) | 100.0% (10/10) |
| exclude | 75.0% (9/12) | 100.0% (12/12) |
| fault | 75.0% (6/8) | 100.0% (8/8) |
| partial | 87.5% (21/24) | 87.5% (21/24) |
| refuse | - | 71.4% (10/14) |
| relative | 87.5% (7/8) | 100.0% (8/8) |
| ticket | 91.7% (22/24) | 91.7% (22/24) |
| time | 90.0% (18/20) | 90.0% (18/20) |
| topk | 57.1% (8/14) | 100.0% (14/14) |

| Style | Exact match |
|---|---|
| chatty | 76.5% (13/17) |
| formal | 77.8% (7/9) |
| keyvalue | 100.0% (3/3) |
| plain | 90.2% (37/41) |
| question | 100.0% (8/8) |
| terse | 90.3% (28/31) |
| typo | 60.0% (3/5) |
| words | 0.0% (0/6) |

Field accuracy on spec items: at 91.0%, exclude_services 90.8%, system 94.2%, ticket 96.4%, top_k 89.2%, want_fault_type 91.7%.

#### keyword/regex parser: failure examples

translation (12 shown):

- `r11b-sealed-004` category=ticket; style=words; text=Ticket one thousand and twelve, online boutique.; gold={'exclude_services': [], 'system': 'online-boutique', 'ticket': 'INC-1012', 'top_k': 3, 'want_fault_type': True}; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'online-boutique'}; status=clarify; reason=Which incident? Give a ticket (INC-nnnn) or a date and time.; routed=incident
- `r11b-sealed-019` category=ticket; style=words; text=Sock shop, ticket one thousand one hundred and four.; gold={'exclude_services': [], 'system': 'sock-shop', 'ticket': 'INC-1104', 'top_k': 3, 'want_fault_type': True}; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'sock-shop'}; status=clarify; reason=Which incident? Give a ticket (INC-nnnn) or a date and time.; routed=incident
- `r11b-sealed-029` category=time; style=words; text=Online boutique, January nineteenth, twenty-seven past seven in the evening.; gold={'at': '--01-19T19:27', 'exclude_services': [], 'system': 'online-boutique', 'top_k': 3, 'want_fault_type': True}; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'online-boutique'}; status=clarify; reason=Which incident? Give a ticket (INC-nnnn) or a date and time.; routed=incident
- `r11b-sealed-039` category=time; style=words; text=Sock shop, twenty past two in the afternoon on 20 January.; gold={'at': '--01-20T14:20', 'exclude_services': [], 'system': 'sock-shop', 'top_k': 3, 'want_fault_type': True}; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'sock-shop', 'at': '--01-20'}; status=clarify; reason=43 sock-shop incidents matching --01-20 at or before 2024-02-22T09:00: INC-1058 (2024-01-20T00:18), INC-1060 (2024-01-20T00:47), INC-1061 (2024-01-20T01:11), INC-1062 (2024-01-20T01:36) and 39 more. Which one?; routed=incident
- `r11b-sealed-050` category=partial; style=chatty; text=Online boutique, on the 19th around a quarter to seven in the evening.; gold={'at': '---19T18:45', 'exclude_services': [], 'system': 'online-boutique', 'top_k': 3, 'want_fault_type': True}; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'online-boutique', 'at': '---19'}; status=clarify; reason=17 online-boutique incidents matching ---19 at or before 2024-02-22T09:00: INC-1027 (2024-01-19T00:18), INC-1028 (2024-01-19T01:01), INC-1029 (2024-01-19T01:30), INC-1030 (2024-01-19T10:32) and 13 more. Which one?; routed=incident
- `r11b-sealed-056` category=partial; style=words; text=Boutique outage on the seventeenth, ten past nine in the evening.; gold={'at': '---17T21:10', 'exclude_services': [], 'system': 'online-boutique', 'top_k': 3, 'want_fault_type': True}; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'online-boutique'}; status=clarify; reason=Which incident? Give a ticket (INC-nnnn) or a date and time.; routed=incident
- `r11b-sealed-058` category=partial; style=typo; text=boutiqe 20th at 13:28 whats wrong; gold={'at': '---20T13:28', 'exclude_services': [], 'system': 'online-boutique', 'top_k': 3, 'want_fault_type': True}; status=refused; reason=no keywords matched; routed=none
- `r11b-sealed-071` category=relative; style=words; text=The online boutique the day before yesterday at about a quarter to midnight.; gold={'at': '2024-02-20T23:45', 'exclude_services': [], 'system': 'online-boutique', 'top_k': 3, 'want_fault_type': True}; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'online-boutique', 'at': '2024-02-20'}; status=ok; routed=incident
- `r11b-sealed-077` category=topk; style=terse; text=boutique INC-1088, just the single most likely culprit.; gold={'exclude_services': [], 'system': 'online-boutique', 'ticket': 'INC-1088', 'top_k': 1, 'want_fault_type': True}; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'online-boutique', 'ticket': 'INC-1088'}; status=ok; routed=incident
- `r11b-sealed-079` category=topk; style=chatty; text=Four likely services for the sock shop failure on the 20th at 21:43, please.; gold={'at': '---20T21:43', 'exclude_services': [], 'system': 'sock-shop', 'top_k': 4, 'want_fault_type': True}; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'sock-shop', 'at': '---20T21:43'}; status=ok; routed=incident
- `r11b-sealed-082` category=topk; style=terse; text=Sock shop, Jan 19 21:22 - top two.; gold={'at': '--01-19T21:22', 'exclude_services': [], 'system': 'sock-shop', 'top_k': 2, 'want_fault_type': True}; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'sock-shop', 'at': '--01-19T21:22'}; status=ok; routed=incident
- `r11b-sealed-083` category=topk; style=plain; text=Boutique INC-1100: name the one most likely service.; gold={'exclude_services': [], 'system': 'online-boutique', 'ticket': 'INC-1100', 'top_k': 1, 'want_fault_type': True}; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'online-boutique', 'ticket': 'INC-1100'}; status=ok; routed=incident

decision (6 shown):

- `r11b-sealed-122` category=clarify; style=terse; text=Could you check INC-1095?; status=refused; reason=Invalid GoalSpec 'rca.investigate': /params: 'system' is a required property; routed=incident
- `r11b-sealed-124` category=clarify; style=chatty; text=Can you check on that outage from last week?; status=refused; reason=Invalid GoalSpec 'rca.investigate': /params: 'system' is a required property; routed=incident
- `r11b-sealed-131` category=refuse; style=chatty; text=I want a ranked list of all 15 services for INC-1060 on the sock shop.; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'sock-shop', 'ticket': 'INC-1060'}; status=ok; routed=incident
- `r11b-sealed-135` category=refuse; style=plain; text=Investigate INC-1073 on the boutique and ignore the inventory-service.; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'online-boutique', 'ticket': 'INC-1073'}; status=ok; routed=incident
- `r11b-sealed-137` category=refuse; style=terse; text=Restart the frontend on the online boutique.; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'online-boutique'}; status=clarify; reason=Which incident? Give a ticket (INC-nnnn) or a date and time.; routed=incident
- `r11b-sealed-139` category=refuse; style=plain; text=Investigate the boutique incident tomorrow at 10:00.; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'online-boutique'}; status=clarify; reason=Which incident? Give a ticket (INC-nnnn) or a date and time.; routed=incident

## Analysis

<!-- analysis:start -->
*Analysis pending the full run.*
<!-- analysis:end -->
