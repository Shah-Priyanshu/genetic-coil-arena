# R11: the ecosystem on real incidents (goal G8)

Generated 2026-10-27 16:23. Command: `python scripts/r11_ecosystem.py report`.

One shared translator picks the domain (squad, plant or incident), extracts the GoalSpec, the command centre runs `rca.investigate` through the G4c trained team on a real RCAEval RE2 incident, the reverse translator summarises the trace, and the summary is routed and translated again.

- **Sealed set:** 130 hand-written incident requests (`experiments/data/r11_sealed_test.jsonl`, SHA-256 `8aeaeb2b899d86eff2453537d1af6ce7f17949c8711fb9d1f46bbdc6ba3e8fad`), plus the R7a sealed spec and clarify items for the routing check (SHA-256 `ca6ff07b5fb66ef21c288ba9a82d5e7763992fe2142b1a582baf597e542b5e01`). Each translator ran once.
- **Catalogue:** 120 test incidents (SS/OB repetitions 2 and 3), fingerprint `409929e01f894a7e24e60398f7825551a502f3e7a6dde1a1f121df1236c60edf`.
- **Team:** the G4c trained metric fault specialist (config {'hidden': 0, 'l2': 1.0, 'epochs': 400}, trained on 150 Train Ticket and SS/OB repetition-1 incidents; leave-one-system-out accuracy OB 70.0%, SS 66.7%, TT 56.7%); through the object layer it gets top-1 88.3% and joint 68.3% on all 120 catalogued incidents (R10 step 3 reported 68.3 joint).

## Verdict

| Translator | G8 verdict | Exact match | Routing | NL - gold task score | Round trip | Fidelity | Warm p95 |
|---|---|---|---|---|---|---|---|
| qwen2.5-coder:14b (refusal policy) | **KILL** | 95.5% (105/110) | 100.0% (273/273) | -3.6 [-7.3, -0.9] | 95.3% (101/106) | 100.0% | 4.40 s |
| gemma4:e4b (refusal policy) | **KILL** | 94.5% (104/110) | 99.6% (272/273) | -4.5 [-9.1, -0.9] | 92.3% (96/104) | 99.3% | 2.50 s |
| keyword/regex parser | **KILL** | 64.5% (71/110) | 98.2% (268/273) | -16.4 [-23.6, -9.1] | n/a | n/a | n/a |

## Per translator

### qwen2.5-coder:14b (refusal policy)

**KILL** (NL task score significantly below the gold run)

| Criterion | Target | Measured | Result |
|---|---|---|---|
| exact match | >= 90% (kill < 85%) | 95.5% (n=110, 95% CI 89.8%-98.0%) | PASS |
| domain routing | >= 98% | 100.0% (n=273, 95% CI 98.6%-100.0%) | PASS |
| task score, NL vs gold | lower CI bound of NL - gold > -5 points | NL 58.2% vs gold 61.8%; diff -3.6 [-7.3, -0.9] points (n=110) | not met |
| round trip | >= 90% | 95.3% (n=106, 95% CI 89.4%-98.0%) | PASS |
| summary field fidelity | >= 90% of facts (kill < 75%) | 100.0% of 769 facts; all facts in 100.0% of 106 summaries | PASS |
| warm end-to-end p95 | <= 10 s | 4.40 s (p50 3.88 s: translate 1.96, team 0.03, summary 1.81) | PASS |
| peak VRAM | <= 12 GB | 9.8 GB | PASS |

Routing by desk: incident 100.0% (118/118); plant 100.0% (76/76); squad 100.0% (79/79).

Final decision on all 130 items (spec answered with a spec, clarify with a question, refuse with a refusal): 93.8% (122/130). clarify: 62.5% (5/8) {'clarify': 5, 'refused': 3}. refuse: 91.7% (11/12) {'ok': 1, 'refused': 11}. spec: 96.4% (106/110) {'ok': 106, 'refused': 4}.

Task score: NL joint accuracy 58.2% (top-1 77.3%), gold 61.8% (top-1 80.9%) over 110 requests; 4 had no run (refused, clarified or invalid) and 0 named a different incident than the gold.

| Category | Exact match | Decision (all kinds) |
|---|---|---|
| clarify | - | 62.5% (5/8) |
| combo | 100.0% (10/10) | 100.0% (10/10) |
| exclude | 100.0% (17/17) | 100.0% (17/17) |
| fault | 100.0% (9/9) | 100.0% (9/9) |
| refuse | - | 91.7% (11/12) |
| relative | 100.0% (7/7) | 100.0% (7/7) |
| ticket | 96.0% (24/25) | 100.0% (25/25) |
| time | 88.0% (22/25) | 88.0% (22/25) |
| topk | 94.1% (16/17) | 94.1% (16/17) |

| Style | Exact match |
|---|---|
| chatty | 90.9% (20/22) |
| formal | 100.0% (8/8) |
| keyvalue | 100.0% (3/3) |
| plain | 97.1% (34/35) |
| question | 100.0% (6/6) |
| terse | 96.0% (24/25) |
| typo | 100.0% (4/4) |
| words | 85.7% (6/7) |

Field accuracy on spec items: at 90.7%, exclude_services 96.4%, system 96.4%, ticket 100.0%, top_k 96.4%, want_fault_type 95.5%.

Fact accuracy: at 100.0%, exclusions 100.0%, fault_probability 100.0%, fault_type 100.0%, onset 100.0%, suspects 100.0%, system 100.0%, ticket 100.0%.

#### qwen2.5-coder:14b (refusal policy): failure examples

translation (5 shown):

- `r11-sealed-040` category=time; style=words; text=Twelve forty-seven at night on the 20th, sock shop. What happened?; gold={'at': '2024-01-20T00:47', 'exclude_services': [], 'system': 'sock-shop', 'top_k': 3, 'want_fault_type': True}; pred={'system': 'sock-shop', 'at': '2024-02-20T23:47', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; status=refused; reason=no sock-shop incident within 10 minutes of 2024-02-20T23:47; routed=incident
- `r11-sealed-063` category=topk; style=terse; text=online boutique on the 17th at 10:20, top 2; gold={'at': '2024-01-17T10:20', 'exclude_services': [], 'system': 'online-boutique', 'top_k': 2, 'want_fault_type': True}; pred={'system': 'online-boutique', 'at': '2024-02-17T10:20', 'top_k': 2, 'want_fault_type': True, 'exclude_services': []}; status=refused; reason=no online-boutique incident within 10 minutes of 2024-02-17T10:20; routed=incident
- `r11-sealed-100` category=ticket; style=plain; text=Online boutique, ticket INC-1045: which service failed?; gold={'exclude_services': [], 'system': 'online-boutique', 'ticket': 'INC-1045', 'top_k': 3, 'want_fault_type': True}; pred={'system': 'online-boutique', 'ticket': 'INC-1045', 'want_fault_type': False, 'top_k': 3, 'exclude_services': []}; status=ok; routed=incident
- `r11-sealed-104` category=time; style=chatty; text=Sock shop again! This was on the 19th, about 21:46. What failed?; gold={'at': '2024-01-19T21:46', 'exclude_services': [], 'system': 'sock-shop', 'top_k': 3, 'want_fault_type': True}; pred={'system': 'sock-shop', 'at': '2024-02-19T21:46', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; status=refused; reason=no sock-shop incident within 10 minutes of 2024-02-19T21:46; routed=incident
- `r11-sealed-109` category=time; style=chatty; text=Around 11:35 on the 17th the boutique had its third problem of the day. Look at that one.; gold={'at': '2024-01-17T11:35', 'exclude_services': [], 'system': 'online-boutique', 'top_k': 3, 'want_fault_type': True}; pred={'system': 'online-boutique', 'at': '2024-02-17T11:35', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; status=refused; reason=no online-boutique incident within 10 minutes of 2024-02-17T11:35; routed=incident

decision (4 shown):

- `r11-sealed-114` category=clarify; style=chatty; text=The shop had an outage on the 20th, look into it.; pred={'system': 'sock-shop', 'at': '2024-02-20', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; status=refused; reason=no sock-shop incident on 2024-02-20; routed=incident
- `r11-sealed-115` category=clarify; style=plain; text=Look into the online boutique outage last week.; pred={'system': 'online-boutique', 'at': '2024-02-15', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; status=refused; reason=no online-boutique incident on 2024-02-15; routed=incident
- `r11-sealed-116` category=clarify; style=terse; text=Investigate an incident.; status=refused; reason=the request is too vague; routed=incident
- `r11-sealed-126` category=refuse; style=plain; text=Online boutique INC-1010, give me the top 9 suspects.; pred={'system': 'online-boutique', 'ticket': 'INC-1010', 'top_k': 5, 'want_fault_type': True, 'exclude_services': []}; status=ok; routed=incident

round_trip (5 shown):

- `r11-sealed-084` summary=Investigation of the sock-shop incident, requested by ticket INC-1074. The suspected services, most likely first, are orders, carts-db; no fault type was requested; onset 06:24 UTC; no services excluded.; recon_status=ok; recon_params={'system': 'sock-shop', 'ticket': 'INC-1074', 'want_fault_type': False, 'top_k': 3, 'exclude_services': []}; gold={'exclude_services': [], 'system': 'sock-shop', 'ticket': 'INC-1074', 'top_k': 2, 'want_fault_type': False}
- `r11-sealed-089` summary=Investigation of the online-boutique incident, requested by ticket INC-1044. The suspected services, most likely first, are currencyservice, frontend, checkoutservice, cartservice, productcatalogservice; no fault type w...; recon_status=ok; recon_params={'system': 'online-boutique', 'ticket': 'INC-1044', 'want_fault_type': False, 'top_k': 3, 'exclude_services': []}; gold={'exclude_services': [], 'system': 'online-boutique', 'ticket': 'INC-1044', 'top_k': 5, 'want_fault_type': False}
- `r11-sealed-092` summary=Investigation of the sock-shop incident, requested by ticket INC-1109. The suspected services, most likely first, are user, shipping; no fault type was requested; onset 21:44 UTC. Excluded: user-db.; recon_status=ok; recon_params={'system': 'sock-shop', 'ticket': 'INC-1109', 'want_fault_type': False, 'exclude_services': ['user-db'], 'top_k': 3}; gold={'exclude_services': ['user-db'], 'system': 'sock-shop', 'ticket': 'INC-1109', 'top_k': 2, 'want_fault_type': False}
- `r11-sealed-093` summary=Investigation of the online-boutique incident, requested by ticket INC-1083. The suspected services, most likely first, are emailservice, checkoutservice; no fault type was requested; onset 09:26 UTC. Excluded: frontend.; recon_status=ok; recon_params={'system': 'online-boutique', 'ticket': 'INC-1083', 'want_fault_type': False, 'exclude_services': ['frontend'], 'top_k': 3}; gold={'exclude_services': ['frontend'], 'system': 'online-boutique', 'ticket': 'INC-1083', 'top_k': 2, 'want_fault_type': False}
- `r11-sealed-096` summary=Investigation of the sock-shop incident, requested by ticket INC-1112. The suspected service is user; no fault type was requested; onset 23:10 UTC; no services excluded.; recon_status=ok; recon_params={'system': 'sock-shop', 'ticket': 'INC-1112', 'want_fault_type': False, 'top_k': 3, 'exclude_services': []}; gold={'exclude_services': [], 'system': 'sock-shop', 'ticket': 'INC-1112', 'top_k': 1, 'want_fault_type': False}

### gemma4:e4b (refusal policy)

**KILL** (NL task score significantly below the gold run)

| Criterion | Target | Measured | Result |
|---|---|---|---|
| exact match | >= 90% (kill < 85%) | 94.5% (n=110, 95% CI 88.6%-97.5%) | PASS |
| domain routing | >= 98% | 99.6% (n=273, 95% CI 98.0%-99.9%) | PASS |
| task score, NL vs gold | lower CI bound of NL - gold > -5 points | NL 57.3% vs gold 61.8%; diff -4.5 [-9.1, -0.9] points (n=110) | not met |
| round trip | >= 90% | 92.3% (n=104, 95% CI 85.6%-96.1%) | PASS |
| summary field fidelity | >= 90% of facts (kill < 75%) | 99.3% of 754 facts; all facts in 95.2% of 104 summaries | PASS |
| warm end-to-end p95 | <= 10 s | 2.50 s (p50 2.26 s: translate 1.18, team 0.04, summary 1.01) | PASS |
| peak VRAM | <= 12 GB | 4.0 GB | PASS |

Routing by desk: incident 100.0% (118/118); plant 98.7% (75/76); squad 100.0% (79/79).

Final decision on all 130 items (spec answered with a spec, clarify with a question, refuse with a refusal): 93.8% (122/130). clarify: 87.5% (7/8) {'clarify': 7, 'refused': 1}. refuse: 91.7% (11/12) {'ok': 1, 'refused': 11}. spec: 94.5% (104/110) {'clarify': 3, 'ok': 104, 'refused': 3}.

Task score: NL joint accuracy 57.3% (top-1 76.4%), gold 61.8% (top-1 80.9%) over 110 requests; 6 had no run (refused, clarified or invalid) and 0 named a different incident than the gold.

| Category | Exact match | Decision (all kinds) |
|---|---|---|
| clarify | - | 87.5% (7/8) |
| combo | 100.0% (10/10) | 100.0% (10/10) |
| exclude | 100.0% (17/17) | 100.0% (17/17) |
| fault | 100.0% (9/9) | 100.0% (9/9) |
| refuse | - | 91.7% (11/12) |
| relative | 71.4% (5/7) | 71.4% (5/7) |
| ticket | 100.0% (25/25) | 100.0% (25/25) |
| time | 88.0% (22/25) | 88.0% (22/25) |
| topk | 94.1% (16/17) | 94.1% (16/17) |

| Style | Exact match |
|---|---|
| chatty | 90.9% (20/22) |
| formal | 100.0% (8/8) |
| keyvalue | 100.0% (3/3) |
| plain | 100.0% (35/35) |
| question | 100.0% (6/6) |
| terse | 88.0% (22/25) |
| typo | 100.0% (4/4) |
| words | 85.7% (6/7) |

Field accuracy on spec items: at 86.0%, exclude_services 94.5%, system 94.5%, ticket 100.0%, top_k 94.5%, want_fault_type 94.5%.

Fact accuracy: at 100.0%, exclusions 95.2%, fault_probability 100.0%, fault_type 100.0%, onset 100.0%, suspects 100.0%, system 100.0%, ticket 100.0%.

#### gemma4:e4b (refusal policy): failure examples

translation (6 shown):

- `r11-sealed-024` category=time; style=terse; text=shop outage jan 20 ~9:40am; gold={'at': '2024-01-20T09:40', 'exclude_services': [], 'system': 'sock-shop', 'top_k': 3, 'want_fault_type': True}; status=clarify; reason=Could you provide a ticket number or a more specific date for the shop outage?; routed=incident
- `r11-sealed-040` category=time; style=words; text=Twelve forty-seven at night on the 20th, sock shop. What happened?; gold={'at': '2024-01-20T00:47', 'exclude_services': [], 'system': 'sock-shop', 'top_k': 3, 'want_fault_type': True}; status=refused; reason=top_k cannot be set to 12 as the maximum allowed value is 5; routed=incident
- `r11-sealed-046` category=relative; style=chatty; text=Yesterday shortly after half past midnight there was a boutique incident. Find the suspects.; gold={'at': '2024-02-21T00:30', 'exclude_services': [], 'system': 'online-boutique', 'top_k': 3, 'want_fault_type': True}; status=clarify; reason=Could you provide a ticket number or a more specific time for the incident?; routed=incident
- `r11-sealed-047` category=relative; style=terse; text=boutique outage on the 20th of February; gold={'at': '2024-02-20', 'exclude_services': [], 'system': 'online-boutique', 'top_k': 3, 'want_fault_type': True}; status=clarify; reason=What year was the outage on February 20th?; routed=incident
- `r11-sealed-063` category=topk; style=terse; text=online boutique on the 17th at 10:20, top 2; gold={'at': '2024-01-17T10:20', 'exclude_services': [], 'system': 'online-boutique', 'top_k': 2, 'want_fault_type': True}; pred={'system': 'online-boutique', 'at': '2024-02-17T10:20', 'top_k': 2, 'want_fault_type': True, 'exclude_services': []}; status=refused; reason=no online-boutique incident within 10 minutes of 2024-02-17T10:20; routed=incident
- `r11-sealed-109` category=time; style=chatty; text=Around 11:35 on the 17th the boutique had its third problem of the day. Look at that one.; gold={'at': '2024-01-17T11:35', 'exclude_services': [], 'system': 'online-boutique', 'top_k': 3, 'want_fault_type': True}; pred={'system': 'online-boutique', 'at': '2024-02-17T11:35', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; status=refused; reason=no online-boutique incident within 10 minutes of 2024-02-17T11:35; routed=incident

decision (2 shown):

- `r11-sealed-116` category=clarify; style=terse; text=Investigate an incident.; status=refused; reason=the request is too vague and does not specify a system, ticket, or time for investigation; routed=incident
- `r11-sealed-127` category=refuse; style=plain; text=Investigate INC-1100 on the boutique but ignore the billing service.; pred={'system': 'online-boutique', 'ticket': 'INC-1100', 'exclude_services': ['paymentservice'], 'top_k': 3, 'want_fault_type': True}; status=ok; routed=incident

round_trip (8 shown):

- `r11-sealed-028` summary=Investigation of the sock-shop incident, requested by time 2024-01-20T07:15 (ticket INC-1076). The suspected services, most likely first, are orders, carts, payment; fault type cpu (probability 0.25); onset 07:17 UTC. E...; recon_status=ok; recon_params={'system': 'sock-shop', 'ticket': 'INC-1076', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; gold={'at': '2024-01-20T07:15', 'exclude_services': [], 'system': 'sock-shop', 'top_k': 3, 'want_fault_type': True}
- `r11-sealed-068` summary=Investigation of the sock-shop incident, requested by time 2024-01-20T10:36 (ticket INC-1086). The suspected services, most likely first, are orders, payment, shipping; fault type delay (probability 0.4); onset 10:36 UT...; recon_status=ok; recon_params={'system': 'sock-shop', 'ticket': 'INC-1086', 'top_k': 3, 'exclude_services': ['front-end'], 'want_fault_type': True}; gold={'at': '2024-01-20T10:36', 'exclude_services': ['front-end'], 'system': 'sock-shop', 'top_k': 3, 'want_fault_type': True}
- `r11-sealed-084` summary=Investigation of the sock-shop incident, requested by ticket INC-1074. The suspected services, most likely first, are orders, carts-db; no fault type was requested; onset 06:24 UTC. Excluded: no services excluded.; recon_status=ok; recon_params={'system': 'sock-shop', 'ticket': 'INC-1074', 'want_fault_type': False, 'top_k': 3, 'exclude_services': []}; gold={'exclude_services': [], 'system': 'sock-shop', 'ticket': 'INC-1074', 'top_k': 2, 'want_fault_type': False}
- `r11-sealed-090` summary=Investigation of the sock-shop incident, requested by time 2024-01-20T19:50 (ticket INC-1105). The suspected services, most likely first, are user; fault type cpu (probability 0.37); onset 19:50 UTC. Excluded: payment.; recon_status=ok; recon_params={'system': 'sock-shop', 'ticket': 'INC-1105', 'top_k': 1, 'want_fault_type': True, 'exclude_services': ['payment']}; gold={'at': '2024-01-20T19:50', 'exclude_services': ['payment'], 'system': 'sock-shop', 'top_k': 1, 'want_fault_type': True}
- `r11-sealed-092` summary=Investigation of the sock-shop incident, requested by ticket INC-1109. The suspected services, most likely first, are user, shipping; no fault type was requested; onset 21:44 UTC. Excluded: user-db.; recon_status=ok; recon_params={'system': 'sock-shop', 'ticket': 'INC-1109', 'want_fault_type': False, 'exclude_services': ['user-db'], 'top_k': 3}; gold={'exclude_services': ['user-db'], 'system': 'sock-shop', 'ticket': 'INC-1109', 'top_k': 2, 'want_fault_type': False}
- `r11-sealed-093` summary=Investigation of the online-boutique incident, requested by ticket INC-1083. The suspected services, most likely first, are emailservice, checkoutservice; no fault type was requested; onset 09:26 UTC. Excluded: frontend.; recon_status=ok; recon_params={'system': 'online-boutique', 'ticket': 'INC-1083', 'want_fault_type': False, 'exclude_services': ['frontend'], 'top_k': 3}; gold={'exclude_services': ['frontend'], 'system': 'online-boutique', 'ticket': 'INC-1083', 'top_k': 2, 'want_fault_type': False}
- `r11-sealed-096` summary=Investigation of the sock-shop incident, requested by ticket INC-1112. The suspected services, most likely first, are user; no fault type was requested; onset 23:10 UTC. Excluded: none.; recon_status=ok; recon_params={'system': 'sock-shop', 'ticket': 'INC-1112', 'want_fault_type': False, 'top_k': 3, 'exclude_services': []}; gold={'exclude_services': [], 'system': 'sock-shop', 'ticket': 'INC-1112', 'top_k': 1, 'want_fault_type': False}
- `r11-sealed-101` summary=Investigation of the sock-shop incident, requested by time 2024-01-20T03:12 (ticket INC-1065). The suspected services, most likely first, are carts, orders, catalogue; fault type mem (probability 0.32); onset 03:12 UTC....; recon_status=ok; recon_params={'system': 'sock-shop', 'ticket': 'INC-1065', 'top_k': 3, 'want_fault_type': True, 'exclude_services': []}; gold={'at': '2024-01-20T03:12', 'exclude_services': [], 'system': 'sock-shop', 'top_k': 3, 'want_fault_type': True}

facts (5 shown):

- `r11-sealed-015` summary=Investigation of the sock-shop incident, requested by ticket INC-1115. The suspected services, most likely first, are front-end, user, orders; fault type loss (probability 0.52); onset 00:30 UTC. Excluded: none.; missing=['exclusions']
- `r11-sealed-056` summary=Investigation of the sock-shop incident, requested by ticket INC-1111. The suspected services, most likely first, are user; fault type socket (probability 0.37); onset 22:44 UTC. Excluded: none.; missing=['exclusions']
- `r11-sealed-060` summary=Investigation of the sock-shop incident, requested by time 2024-01-19T22:11 (ticket INC-1050). The suspected services, most likely first, are catalogue, orders, front-end, user; fault type mem (probability 0.4); onset 2...; missing=['exclusions']
- `r11-sealed-082` summary=Investigation of the sock-shop incident, requested by ticket INC-1104. The suspected services, most likely first, are user, orders, payment; no fault type was requested; onset 19:23 UTC. Excluded: none.; missing=['exclusions']
- `r11-sealed-096` summary=Investigation of the sock-shop incident, requested by ticket INC-1112. The suspected services, most likely first, are user; no fault type was requested; onset 23:10 UTC. Excluded: none.; missing=['exclusions']

### keyword/regex parser

**KILL** (exact match 64.5% < 85%; NL task score significantly below the gold run)

| Criterion | Target | Measured | Result |
|---|---|---|---|
| exact match | >= 90% (kill < 85%) | 64.5% (n=110, 95% CI 55.3%-72.9%) | not met |
| domain routing | >= 98% | 98.2% (n=273, 95% CI 95.8%-99.2%) | PASS |
| task score, NL vs gold | lower CI bound of NL - gold > -5 points | NL 45.5% vs gold 61.8%; diff -16.4 [-23.6, -9.1] points (n=110) | not met |
| round trip | >= 90% | n/a | n/a |
| summary field fidelity | >= 90% of facts (kill < 75%) | n/a | n/a |
| warm end-to-end p95 | <= 10 s | n/a | n/a |
| peak VRAM | <= 12 GB | n/a | n/a |

Routing by desk: incident 100.0% (118/118); plant 98.7% (75/76); squad 94.9% (75/79).

Final decision on all 130 items (spec answered with a spec, clarify with a question, refuse with a refusal): 77.7% (101/130). clarify: 62.5% (5/8) {'clarify': 5, 'refused': 3}. refuse: 75.0% (9/12) {'clarify': 1, 'ok': 2, 'refused': 9}. spec: 79.1% (87/110) {'clarify': 14, 'ok': 87, 'refused': 9}.

Task score: NL joint accuracy 45.5% (top-1 63.6%), gold 61.8% (top-1 80.9%) over 110 requests; 23 had no run (refused, clarified or invalid) and 0 named a different incident than the gold.

| Category | Exact match | Decision (all kinds) |
|---|---|---|
| clarify | - | 62.5% (5/8) |
| combo | 60.0% (6/10) | 100.0% (10/10) |
| exclude | 52.9% (9/17) | 88.2% (15/17) |
| fault | 88.9% (8/9) | 100.0% (9/9) |
| refuse | - | 75.0% (9/12) |
| relative | 42.9% (3/7) | 42.9% (3/7) |
| ticket | 88.0% (22/25) | 88.0% (22/25) |
| time | 56.0% (14/25) | 56.0% (14/25) |
| topk | 52.9% (9/17) | 82.4% (14/17) |

| Style | Exact match |
|---|---|
| chatty | 50.0% (11/22) |
| formal | 75.0% (6/8) |
| keyvalue | 66.7% (2/3) |
| plain | 71.4% (25/35) |
| question | 66.7% (4/6) |
| terse | 80.0% (20/25) |
| typo | 50.0% (2/4) |
| words | 14.3% (1/7) |

Field accuracy on spec items: at 60.5%, exclude_services 70.9%, system 79.1%, ticket 91.0%, top_k 73.6%, want_fault_type 76.4%.

#### keyword/regex parser: failure examples

translation (12 shown):

- `r11-sealed-005` category=ticket; style=words; text=Ticket one thousand sixty-eight, sock shop.; gold={'exclude_services': [], 'system': 'sock-shop', 'ticket': 'INC-1068', 'top_k': 3, 'want_fault_type': True}; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'sock-shop'}; status=clarify; reason=Which incident? Give a ticket (INC-nnnn) or a date and time.; routed=incident
- `r11-sealed-008` category=ticket; style=typo; text=investgate INC-1033 on teh online boutiqe; gold={'exclude_services': [], 'system': 'online-boutique', 'ticket': 'INC-1033', 'top_k': 3, 'want_fault_type': True}; status=refused; reason=Invalid GoalSpec 'rca.investigate': /params: 'system' is a required property; routed=incident
- `r11-sealed-020` category=ticket; style=plain; text=Let's have a look at incident 1088 in the boutique system.; gold={'exclude_services': [], 'system': 'online-boutique', 'ticket': 'INC-1088', 'top_k': 3, 'want_fault_type': True}; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'online-boutique'}; status=clarify; reason=Which incident? Give a ticket (INC-nnnn) or a date and time.; routed=incident
- `r11-sealed-026` category=time; style=words; text=The sock shop failure at 11:04pm on the 19th of January.; gold={'at': '2024-01-19T23:04', 'exclude_services': [], 'system': 'sock-shop', 'top_k': 3, 'want_fault_type': True}; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'sock-shop', 'at': '2024-01-19T11:04'}; status=refused; reason=no sock-shop incident within 10 minutes of 2024-01-19T11:04; routed=incident
- `r11-sealed-027` category=time; style=plain; text=Look at the boutique's incident at 3:50 pm on Jan 18th.; gold={'at': '2024-01-18T15:50', 'exclude_services': [], 'system': 'online-boutique', 'top_k': 3, 'want_fault_type': True}; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'online-boutique', 'at': '2024-01-18T03:50'}; status=refused; reason=no online-boutique incident within 10 minutes of 2024-01-18T03:50; routed=incident
- `r11-sealed-030` category=time; style=words; text=sock shop incident Sunday 21st January at 2:48 in the afternoon; gold={'at': '2024-01-21T14:48', 'exclude_services': [], 'system': 'sock-shop', 'top_k': 3, 'want_fault_type': True}; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'sock-shop', 'at': '2024-01-21T02:48'}; status=refused; reason=no sock-shop incident within 10 minutes of 2024-01-21T02:48; routed=incident
- `r11-sealed-033` category=time; style=terse; text=Online boutique 17/01/2024 21:50 incident - root cause?; gold={'at': '2024-01-17T21:50', 'exclude_services': [], 'system': 'online-boutique', 'top_k': 3, 'want_fault_type': True}; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'online-boutique'}; status=clarify; reason=Which incident? Give a ticket (INC-nnnn) or a date and time.; routed=incident
- `r11-sealed-036` category=time; style=words; text=investigate shop outage at about 5am on January 20th; gold={'at': '2024-01-20T05:00', 'exclude_services': [], 'system': 'sock-shop', 'top_k': 3, 'want_fault_type': True}; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'sock-shop', 'at': '2024-01-20'}; status=clarify; reason=43 sock-shop incidents on 2024-01-20: INC-1058 (2024-01-20T00:18), INC-1060 (2024-01-20T00:47), INC-1061 (2024-01-20T01:11), INC-1062 (2024-01-20T01:36) and 39 more. Which one?; routed=incident
- `r11-sealed-037` category=time; style=keyvalue; text=boutique 10:30 on 2024-01-19; gold={'at': '2024-01-19T10:30', 'exclude_services': [], 'system': 'online-boutique', 'top_k': 3, 'want_fault_type': True}; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'online-boutique', 'at': '2024-01-19'}; status=clarify; reason=17 online-boutique incidents on 2024-01-19: INC-1027 (2024-01-19T00:18), INC-1028 (2024-01-19T01:01), INC-1029 (2024-01-19T01:30), INC-1030 (2024-01-19T10:32) and 13 more. Which one?; routed=incident
- `r11-sealed-039` category=time; style=question; text=What broke the online boutique at 5:20 pm on 17 January?; gold={'at': '2024-01-17T17:20', 'exclude_services': [], 'system': 'online-boutique', 'top_k': 3, 'want_fault_type': True}; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'online-boutique', 'at': '2024-01-17T05:20'}; status=refused; reason=no online-boutique incident within 10 minutes of 2024-01-17T05:20; routed=incident
- `r11-sealed-040` category=time; style=words; text=Twelve forty-seven at night on the 20th, sock shop. What happened?; gold={'at': '2024-01-20T00:47', 'exclude_services': [], 'system': 'sock-shop', 'top_k': 3, 'want_fault_type': True}; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'sock-shop'}; status=clarify; reason=Which incident? Give a ticket (INC-nnnn) or a date and time.; routed=incident
- `r11-sealed-042` category=relative; style=chatty; text=Yesterday around 04:20 the boutique had an incident, look into it.; gold={'at': '2024-02-21T04:20', 'exclude_services': [], 'system': 'online-boutique', 'top_k': 3, 'want_fault_type': True}; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'online-boutique'}; status=clarify; reason=Which incident? Give a ticket (INC-nnnn) or a date and time.; routed=incident

decision (6 shown):

- `r11-sealed-110` category=clarify; style=plain; text=Investigate INC-1042.; status=refused; reason=Invalid GoalSpec 'rca.investigate': /params: 'system' is a required property; routed=incident
- `r11-sealed-111` category=clarify; style=terse; text=Please look at ticket 1090; status=refused; reason=Invalid GoalSpec 'rca.investigate': /params: 'system' is a required property; routed=incident
- `r11-sealed-116` category=clarify; style=terse; text=Investigate an incident.; status=refused; reason=Invalid GoalSpec 'rca.investigate': /params: 'system' is a required property; routed=incident
- `r11-sealed-123` category=refuse; style=terse; text=Restart the cart service on the sock shop.; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'sock-shop'}; status=clarify; reason=Which incident? Give a ticket (INC-nnnn) or a date and time.; routed=incident
- `r11-sealed-125` category=refuse; style=plain; text=Investigate INC-1020 and also INC-1021 on the online boutique, as one request.; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'online-boutique', 'ticket': 'INC-1020'}; status=ok; routed=incident
- `r11-sealed-127` category=refuse; style=plain; text=Investigate INC-1100 on the boutique but ignore the billing service.; pred={'top_k': 3, 'want_fault_type': True, 'exclude_services': [], 'system': 'online-boutique', 'ticket': 'INC-1100'}; status=ok; routed=incident

## Analysis

<!-- analysis:start -->
- **G8 is a kill by its pre-registered rule, on one criterion.** For both LLM translators the task score from natural language is significantly below the gold-spec run: 14B −3.6 [−7.3, −0.9], gemma4:e4b −4.5 [−9.1, −0.9]. Every other criterion passes with margin.
  - 14B: exact match 95.5% [89.8, 98.0]; routing 100% (273/273) across the squad, plant and incident domains; round trip 95.3%; summary fidelity 100% of 769 facts; warm p95 4.4 s; 9.8 GB.
  - gemma4:e4b at 4.0 GB is close behind: 94.5%, 99.6%, 92.3%, 99.3%, p95 2.5 s.
  - The keyword parser fails every criterion it can be scored on (64.5% exact, −16.4 task score). That number is optimistic, because the parser was written after the sealed set.
- **The trained team behaves the same through the object layer.** Through the command centre with signed envelopes it scores 68.3 joint on all 120 incidents, exactly R10 step 3's number. It takes 0.03 s per incident; the 4 s of latency is translation (2.0 s) plus summary (1.8 s).
- **The whole 14B gap comes from four requests with a day but no month.**
  - Items 040, 063, 104 and 109 say "on the 17th", "on the 19th" or "on the 20th". The gold labels are January dates; the desk clock is Thursday 22 February 2024.
  - The 14B read each as the most recent such day, in February. No incident lay within 10 minutes of that time, so the desk refused and the item scored 0. Item 040 also has a real error: "twelve forty-seven at night" became 23:47 instead of 00:47.
  - On the other 106 spec items the NL run scores exactly what the gold run scores, and no request was resolved to the wrong incident.
  - gemma4:e4b's five lost items are three of the same month-less dates (040, 063 and 109, refused), plus two dated requests it answered with a question instead of a spec (024 and 046). It also asked "what year?" for "the 20th of February" (047), an item the gold run scored 0 on anyway.
- **The labels on these four items are the flaw, not the translator.** Given the desk clock, "the 17th" naturally means 17 February. The gold needs to know that the incidents happened in January, and only the catalogue knows that. These items should have carried a month, or been clarify items. Changing the desk now and rescoring the same sealed set would be tuning on it, so the kill stands.
- **The criterion is strict, as intended.** At n = 110, the lower bound > −5 tolerates about two dropped requests that gold got right. Four month-less dates were enough to fail it.
- **Smaller failures to fix in a v2:**
  - Clarify: the 14B asked a question in only 5 of 8 vague requests and refused the other 3 (e.g. "Investigate an incident.").
  - Refuse: it clamped "top 9 suspects" to 5 instead of refusing.
  - Round trip: all five failures lose `top_k`. When the team returns k suspects the summary lists them but never states that k were requested, so re-translation falls back to the default 3. The summary should state the requested k.
- **What G8 shows.**
  - The ecosystem runs end to end on real incident data: plain request → shared translator → command centre → trained specialist team → trace → summary → re-translation.
  - Translation, routing, round trip, fidelity, latency and memory all meet the targets set in advance.
  - The weak point is resolving loose time references against a catalogue. The desk should resolve a day of the month to the most recent month that has a matching incident, or ask.
  - gemma4:e4b at 4 GB nearly matches the 14B, which leaves room on the card for specialists.
- **Next (R11 v2):**
  1. Fix the desk's date resolution and state top-k in the summary.
  2. Have someone other than the author write a fresh sealed set, with every gold checked against the desk clock for a unique natural reading.
  3. Rerun. About 20 GPU minutes per translator.
<!-- analysis:end -->
