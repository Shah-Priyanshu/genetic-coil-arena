# R12c: the desk checks the translator's literals (goal G10b)

Generated 2026-10-28 00:48. Command: `python scripts/r12c_desk.py report`.

Built locally from the `r12c-*-sealed` job artifacts (pinned dbd41b46, code of 747058f4) with the post-hoc rows from `r12-o14-sealed` and `r11b-o14-sealed`, the same inputs as the runner's `r12c-zz-aggregate`.

- **Sealed set:** 106 hand-written requests (`experiments/data/r12c_sealed_test.jsonl`, SHA-256 `e20d15a588c6b0f35be26d668e9b9817ce1ea7147fd436ac55fd4af00726968f`), 13 of them guess traps. Each translator ran once; the report scores the same raw translations with the literal check off (the G10 desk) and on (desk v4).
- **Catalogue:** 182 test incidents, fingerprint `12e4ce9fd1d393ee7c01092ae35572b28b4d73d0e33357327075e186a9bd8952`; desk clock Wednesday 2024-12-11 09:00 UTC; prompts and spec v3 as in G10.
- **Rules** (`envs/logtriage/literal_check.py`): relative (a relative day in the text sets the date from the desk clock), year (a year the text does not name is dropped), month (a month the text does not name is dropped), top_k (the one suspect count the text states replaces a different top_k).

### Verdict

| Translator | G10b verdict | Exact off | Exact on | Fixed / broken | McNemar p | Silent wrong (off / on) | NL - gold, on |
|---|---|---|---|---|---|---|---|
| qwen2.5-coder:14b | **NOT PASSED** | 84.5% (71/84) | 97.6% (82/84) | 11 / 0 | 0.0010 | 5 / 0 | -2.4 [-6.0, +0.0] |
| gemma4:e4b | **NOT PASSED** | 76.2% (64/84) | 90.5% (76/84) | 12 / 0 | 0.0005 | 0 / 0 | -3.6 [-8.3, +0.0] |
| keyword/regex parser | **NOT PASSED** | 90.5% (76/84) | 92.9% (78/84) | 2 / 0 | 0.5000 | 4 / 1 | -3.6 [-8.3, +0.0] |

### Per translator

### qwen2.5-coder:14b

**NOT PASSED**

| Criterion | Target | Measured | Result |
|---|---|---|---|
| exact match, check on | >= 95% (kill < 90%) | 97.6% (n=84, 95% CI 91.7%-99.3%) | PASS |
| silent wrong answers, check on | <= 1 | 0 | PASS |
| exact match, check on - check off | >= +3 points, exact binomial p < 0.05 | +13.1 points (11 fixed, 0 broken), p = 0.0010 | PASS |
| task score, NL vs gold, check on | lower CI bound of NL - gold > -5 points | NL 58.3% vs gold 60.7%; diff -2.4 [-6.0, +0.0] points (n=84) | not met |

| | Check off (G10 desk) | Check on (desk v4) |
|---|---|---|
| exact match, spec items | 84.5% (n=84, 95% CI 75.3%-90.7%) | 97.6% (n=84, 95% CI 91.7%-99.3%) |
| silent wrong answers | 5 | 0 |
| silent wrong on the guess traps | 5 | 0 |
| right incident, other field wrong | 10 | 0 |
| decision (spec / clarify / refuse) | 92.5% (98/106) | 98.1% (104/106) |
| NL task score vs gold (mean diff) | -2.4 [-6.0, +0.0] | -2.4 [-6.0, +0.0] |

The check changed 18 of 106 translations; rules fired: month 2, top_k 5, year 11.

| Category | Exact off | Exact on |
|---|---|---|
| combo | 66.7% (4/6) | 66.7% (4/6) |
| exclude | 83.3% (5/6) | 100.0% (6/6) |
| fault | 75.0% (3/4) | 100.0% (4/4) |
| partial | 100.0% (2/2) | 100.0% (2/2) |
| relative | 100.0% (8/8) | 100.0% (8/8) |
| ticket | 100.0% (8/8) | 100.0% (8/8) |
| time | 76.3% (29/38) | 100.0% (38/38) |
| topk | 100.0% (12/12) | 100.0% (12/12) |

Items the check fixed, broke or left wrong:

- *fixed* `r12c-sealed-008` "Online boutique, 22 Nov at 12:24.": gold at=--11-22T12:24, top_k=3; raw at=2024-11-22T12:24, top_k=3 (ok); checked at=--11-22T12:24, top_k=3 (ok); rules ['year']
- *fixed* `r12c-sealed-010` "train ticket, Dec 8th, 07:28": gold at=--12-08T07:28, top_k=3; raw at=2024-12-08T07:28, top_k=3 (ok); checked at=--12-08T07:28, top_k=3 (ok); rules ['year']
- *fixed* `r12c-sealed-014` "Train ticket system, 8 December at 3:44 pm.": gold at=--12-08T15:44, top_k=3; raw at=2024-12-08T15:44, top_k=3 (ok); checked at=--12-08T15:44, top_k=3 (ok); rules ['year']
- *fixed* `r12c-sealed-016` "shop nov 24 13:34 top 2": gold at=--11-24T13:34, top_k=2; raw at=2024-11-24T13:34, top_k=2 (ok); checked at=--11-24T13:34, top_k=2 (ok); rules ['year']
- *fixed* `r12c-sealed-018` "Train ticket, December 10, 13:14.": gold at=--12-10T13:14, top_k=3; raw at=2024-12-10T13:14, top_k=3 (ok); checked at=--12-10T13:14, top_k=3 (ok); rules ['year']
- *fixed* `r12c-sealed-019` "Boutique went sideways on 23 Nov around 14:47, give me four suspects.": gold at=--11-23T14:47, top_k=4; raw at=2024-11-23T14:47, top_k=4 (ok); checked at=--11-23T14:47, top_k=4 (ok); rules ['year']
- *fixed* `r12c-sealed-021` "Can you take the train ticket problem of Dec 10 at twelve thirty-eight?": gold at=--12-10T12:38, top_k=3; raw at=2024-12-10T12:38, top_k=3 (ok); checked at=--12-10T12:38, top_k=3 (ok); rules ['year']
- *fixed* `r12c-sealed-023` "Sock shop on 22nd November, 6:34 am.": gold at=--11-22T06:34, top_k=3; raw at=2024-11-22T06:34, top_k=3 (ok); checked at=--11-22T06:34, top_k=3 (ok); rules ['year']
- *fixed* `r12c-sealed-029` "I would appreciate an analysis of the Online Boutique incident on the 19th at 14:16.": gold at=---19T14:16, top_k=3; raw at=--01-19T14:16, top_k=3 (ok); checked at=---19T14:16, top_k=3 (ok); rules ['month']
- *fixed* `r12c-sealed-071` "Boutique 23 Nov 12:39 without frontend.": gold at=--11-23T12:39, top_k=3, excl=['frontend']; raw at=2023-11-23T12:39, top_k=3, excl=['frontend'] (refused); checked at=--11-23T12:39, top_k=3, excl=['frontend'] (ok); rules ['year']
- *fixed* `r12c-sealed-076` "Train ticket, Dec 8 at 09:50 - skip the fault type, I only want names.": gold at=--12-08T09:50, top_k=3, no fault type; raw at=2024-12-08T09:50, top_k=3, no fault type (ok); checked at=--12-08T09:50, top_k=3, no fault type (ok); rules ['year']
- *still wrong* `r12c-sealed-078` "Boutique on the 19th at 20:47: five suspects, ignore redis, no fault type.": gold at=---19T20:47, top_k=5, excl=['redis'], no fault type; raw at=--01-19T20:47, top_k=5, excl=['redis', 'want_fault_type'] (refused); checked at=---19T20:47, top_k=5, excl=['redis', 'want_fault_type'] (refused); rules ['month']
- *still wrong* `r12c-sealed-081` "INC-2027 on the Online Boutique: two suspects, without cartservice, no fault type.": gold ticket=INC-2027, top_k=2, excl=['cartservice'], no fault type; raw - (refused); checked - (refused); rules -

### gemma4:e4b

**NOT PASSED**

| Criterion | Target | Measured | Result |
|---|---|---|---|
| exact match, check on | >= 95% (kill < 90%) | 90.5% (n=84, 95% CI 82.3%-95.1%) | not met |
| silent wrong answers, check on | <= 1 | 0 | PASS |
| exact match, check on - check off | >= +3 points, exact binomial p < 0.05 | +14.3 points (12 fixed, 0 broken), p = 0.0005 | PASS |
| task score, NL vs gold, check on | lower CI bound of NL - gold > -5 points | NL 57.1% vs gold 60.7%; diff -3.6 [-8.3, +0.0] points (n=84) | not met |

| | Check off (G10 desk) | Check on (desk v4) |
|---|---|---|
| exact match, spec items | 76.2% (n=84, 95% CI 66.1%-84.0%) | 90.5% (n=84, 95% CI 82.3%-95.1%) |
| silent wrong answers | 0 | 0 |
| silent wrong on the guess traps | 0 | 0 |
| right incident, other field wrong | 9 | 0 |
| decision (spec / clarify / refuse) | 88.7% (94/106) | 91.5% (97/106) |
| NL task score vs gold (mean diff) | -6.0 [-11.9, -1.2] | -3.6 [-8.3, +0.0] |

The check changed 14 of 106 translations; rules fired: relative 3, year 11.

| Category | Exact off | Exact on |
|---|---|---|
| combo | 83.3% (5/6) | 100.0% (6/6) |
| exclude | 83.3% (5/6) | 100.0% (6/6) |
| fault | 50.0% (2/4) | 100.0% (4/4) |
| partial | 50.0% (1/2) | 50.0% (1/2) |
| relative | 75.0% (6/8) | 100.0% (8/8) |
| ticket | 100.0% (8/8) | 100.0% (8/8) |
| time | 68.4% (26/38) | 81.6% (31/38) |
| topk | 91.7% (11/12) | 100.0% (12/12) |

Items the check fixed, broke or left wrong:

- *fixed* `r12c-sealed-008` "Online boutique, 22 Nov at 12:24.": gold at=--11-22T12:24, top_k=3; raw at=2024-11-22T12:24, top_k=3 (ok); checked at=--11-22T12:24, top_k=3 (ok); rules ['year']
- *fixed* `r12c-sealed-009` "Please check the sock shop incident of November 23rd at 10:26.": gold at=--11-23T10:26, top_k=3; raw at=2024-11-23T10:26, top_k=3 (ok); checked at=--11-23T10:26, top_k=3 (ok); rules ['year']
- *fixed* `r12c-sealed-010` "train ticket, Dec 8th, 07:28": gold at=--12-08T07:28, top_k=3; raw at=2024-12-08T07:28, top_k=3 (ok); checked at=--12-08T07:28, top_k=3 (ok); rules ['year']
- *fixed* `r12c-sealed-016` "shop nov 24 13:34 top 2": gold at=--11-24T13:34, top_k=2; raw at=2024-11-24T13:34, top_k=2 (ok); checked at=--11-24T13:34, top_k=2 (ok); rules ['year']
- *fixed* `r12c-sealed-019` "Boutique went sideways on 23 Nov around 14:47, give me four suspects.": gold at=--11-23T14:47, top_k=4; raw at=2023-11-23T14:47, top_k=4 (refused); checked at=--11-23T14:47, top_k=4 (ok); rules ['year']
- *fixed* `r12c-sealed-051` "Look at the train ticket incident from three days ago, 10:26.": gold at=2024-12-08T10:26, top_k=3; raw at=2024-12-09T10:26, top_k=3 (refused); checked at=2024-12-08T10:26, top_k=3 (ok); rules ['relative']
- *fixed* `r12c-sealed-055` "Train ticket, 3 days ago, 8:39 am, no fault type.": gold at=2024-12-08T08:39, top_k=3, no fault type; raw at=2024-12-09T08:39, top_k=3, no fault type (refused); checked at=2024-12-08T08:39, top_k=3, no fault type (ok); rules ['relative']
- *fixed* `r12c-sealed-066` "Boutique, Nov 22 at 11:33, the three most likely suspects.": gold at=--11-22T11:33, top_k=3; raw at=2024-11-22T11:33, top_k=3 (ok); checked at=--11-22T11:33, top_k=3 (ok); rules ['year']
- *fixed* `r12c-sealed-071` "Boutique 23 Nov 12:39 without frontend.": gold at=--11-23T12:39, top_k=3, excl=['frontend']; raw at=2024-11-23T12:39, top_k=3, excl=['frontend'] (ok); checked at=--11-23T12:39, top_k=3, excl=['frontend'] (ok); rules ['year']
- *fixed* `r12c-sealed-075` "suspects only, sock shop, 20 Jan 12:29": gold at=--01-20T12:29, top_k=3, no fault type; raw at=2024-01-20T12:29, top_k=3, no fault type (ok); checked at=--01-20T12:29, top_k=3, no fault type (ok); rules ['year']
- *fixed* `r12c-sealed-076` "Train ticket, Dec 8 at 09:50 - skip the fault type, I only want names.": gold at=--12-08T09:50, top_k=3, no fault type; raw at=2024-12-08T09:50, top_k=3, no fault type (ok); checked at=--12-08T09:50, top_k=3, no fault type (ok); rules ['year']
- *fixed* `r12c-sealed-080` "sock shop Nov 23 20:43, top 4, leave out carts-db": gold at=--11-23T20:43, top_k=4, excl=['carts-db']; raw at=2024-11-23T20:43, top_k=4, excl=['carts-db'] (ok); checked at=--11-23T20:43, top_k=4, excl=['carts-db'] (ok); rules ['year']
- *still wrong* `r12c-sealed-012` "sock shop jan 20 4:58 am": gold at=--01-20T04:58, top_k=3; raw - (clarify); checked - (clarify); rules -
- *still wrong* `r12c-sealed-014` "Train ticket system, 8 December at 3:44 pm.": gold at=--12-08T15:44, top_k=3; raw - (refused); checked - (refused); rules -
- *still wrong* `r12c-sealed-018` "Train ticket, December 10, 13:14.": gold at=--12-10T13:14, top_k=3; raw - (clarify); checked - (clarify); rules -
- *still wrong* `r12c-sealed-021` "Can you take the train ticket problem of Dec 10 at twelve thirty-eight?": gold at=--12-10T12:38, top_k=3; raw at=2024-12-10T00:38, top_k=3 (refused); checked at=--12-10T00:38, top_k=3 (refused); rules ['year']
- *still wrong* `r12c-sealed-023` "Sock shop on 22nd November, 6:34 am.": gold at=--11-22T06:34, top_k=3; raw - (refused); checked - (refused); rules -
- *still wrong* `r12c-sealed-024` "The online boutique incident of 20 February.": gold at=--02-20, top_k=3; raw - (clarify); checked - (clarify); rules -
- *still wrong* `r12c-sealed-029` "I would appreciate an analysis of the Online Boutique incident on the 19th at 14:16.": gold at=---19T14:16, top_k=3; raw - (refused); checked - (refused); rules -
- *still wrong* `r12c-sealed-030` "train ticket 10th 12:03": gold at=---10T12:03, top_k=3; raw - (clarify); checked - (clarify); rules -

### keyword/regex parser

**NOT PASSED**

| Criterion | Target | Measured | Result |
|---|---|---|---|
| exact match, check on | >= 95% (kill < 90%) | 92.9% (n=84, 95% CI 85.3%-96.7%) | not met |
| silent wrong answers, check on | <= 1 | 1 (r12c-sealed-104) | PASS |
| exact match, check on - check off | >= +3 points, exact binomial p < 0.05 | +2.4 points (2 fixed, 0 broken), p = 0.5000 | not met |
| task score, NL vs gold, check on | lower CI bound of NL - gold > -5 points | NL 57.1% vs gold 60.7%; diff -3.6 [-8.3, +0.0] points (n=84) | not met |

| | Check off (G10 desk) | Check on (desk v4) |
|---|---|---|
| exact match, spec items | 90.5% (n=84, 95% CI 82.3%-95.1%) | 92.9% (n=84, 95% CI 85.3%-96.7%) |
| silent wrong answers | 4 | 1 |
| silent wrong on the guess traps | 3 | 0 |
| right incident, other field wrong | 3 | 1 |
| decision (spec / clarify / refuse) | 90.6% (96/106) | 93.4% (99/106) |
| NL task score vs gold (mean diff) | -3.6 [-8.3, +0.0] | -3.6 [-8.3, +0.0] |

The check changed 5 of 106 translations; rules fired: top_k 5.

| Category | Exact off | Exact on |
|---|---|---|
| combo | 100.0% (6/6) | 100.0% (6/6) |
| exclude | 100.0% (6/6) | 100.0% (6/6) |
| fault | 75.0% (3/4) | 75.0% (3/4) |
| partial | 100.0% (2/2) | 100.0% (2/2) |
| relative | 50.0% (4/8) | 50.0% (4/8) |
| ticket | 100.0% (8/8) | 100.0% (8/8) |
| time | 97.4% (37/38) | 97.4% (37/38) |
| topk | 83.3% (10/12) | 100.0% (12/12) |

Items the check fixed, broke or left wrong:

- *fixed* `r12c-sealed-059` "For the sock shop incident INC-1060 I want the 5 most likely culprits.": gold ticket=INC-1060, top_k=5; raw top_k=3, ticket=INC-1060 (ok); checked top_k=5, ticket=INC-1060 (ok); rules ['top_k']
- *fixed* `r12c-sealed-062` "Top five please: sock shop, INC-2019.": gold ticket=INC-2019, top_k=5; raw top_k=3, ticket=INC-2019 (ok); checked top_k=5, ticket=INC-2019 (ok); rules ['top_k']
- *still wrong* `r12c-sealed-021` "Can you take the train ticket problem of Dec 10 at twelve thirty-eight?": gold at=--12-10T12:38, top_k=3; raw top_k=3, at=--12-10 (clarify); checked top_k=3, at=--12-10 (clarify); rules -
- *still wrong* `r12c-sealed-051` "Look at the train ticket incident from three days ago, 10:26.": gold at=2024-12-08T10:26, top_k=3; raw top_k=3 (clarify); checked top_k=3 (clarify); rules -
- *still wrong* `r12c-sealed-052` "3 days ago at 14:33, train ticket system.": gold at=2024-12-08T14:33, top_k=3; raw top_k=3 (clarify); checked top_k=3 (clarify); rules -
- *still wrong* `r12c-sealed-053` "Three days ago around 06:18 the train ticket platform fell over, can you check?": gold at=2024-12-08T06:18, top_k=3; raw top_k=3 (clarify); checked top_k=3 (clarify); rules -
- *still wrong* `r12c-sealed-055` "Train ticket, 3 days ago, 8:39 am, no fault type.": gold at=2024-12-08T08:39, top_k=3, no fault type; raw top_k=3, no fault type (clarify); checked top_k=3, no fault type (clarify); rules -
- *still wrong* `r12c-sealed-075` "suspects only, sock shop, 20 Jan 12:29": gold at=--01-20T12:29, top_k=3, no fault type; raw top_k=3, at=--01-20T12:29 (ok); checked top_k=3, at=--01-20T12:29 (ok); rules -
- *silent wrong* `r12c-sealed-104` "Online boutique INC-1020, ignore the service basket-service.": gold -; raw top_k=3, ticket=INC-1020 (ok); checked top_k=3, ticket=INC-1020 (ok); rules -

### Post-hoc (not part of any verdict)

The check applied to the stored raw predictions of the earlier sealed 14B runs (exact match re-scored from the stored params; resolution and the team are not re-run).

| Earlier run | Spec items | Exact as run | Exact with the check | Fixed / broken | Answered items the check would refuse |
|---|---|---|---|---|---|
| r12-o14-sealed | 91 | 87.9% (80/91) | 97.8% (89/91) | 9 / 0 | 0 |
| r11b-o14-sealed | 120 | 92.5% (111/120) | 97.5% (117/120) | 6 / 0 | 1 |

## Analysis

<!-- analysis:start -->
**G10b is not passed, but the literal check does what it was built to do.** With qwen2.5-coder:14b, three of the four criteria pass with the check on:

| Criterion | Bar | Result |
|---|---|---|
| Exact match, check on | ≥ 95% | 97.6% [91.7, 99.3], up from 84.5% with the check off |
| Paired effect of the check | ≥ +3 points, p < 0.05 | +13.1 points: 11 fixed, 0 broken, exact binomial p = 0.001 |
| Silent wrong answers | ≤ 1 | 0, down from 5 with the check off |
| NL − gold task score | lower bound > −5 | −2.4 [−6.0, +0.0]: not met |

- **Silent wrong answers:** all five with the check off were refusal traps. Each asked for 6 to 12 suspects; the model wrote 5 and the desk answered. With the check on, the text's count reaches validation and is refused. The model asked about every ambiguous online-boutique "20th" itself, so the month traps never fired.
- **Decisions:** 104 of 106 right, against 98 with the check off.

**The task-score miss is identical with the check off, and the check does not touch it.** It comes from 2 of 84 spec items, both requests that combine an exclusion with "no fault type":
- **sealed-078,** "ignore redis, no fault type": the model put `want_fault_type` into the exclusion list, and the desk refused because no service has that name. This is the third sealed set where the 14B makes exactly this error (G8b sealed-116, G10 sealed-089).
- **sealed-081,** "without cartservice, no fault type": the model did not answer within the backend's 600 s timeout. The dev set's one hang had the same shape ("leave out ts-station-service, no fault type").

The translator sets no token cap, so this is most likely a runaway structured generation on that pattern. That is a hypothesis; the raw generation was not captured.

**Which rules fired:** the year rule 11 times, top_k 5 times, the month rule twice. It changed 18 of 106 translations and broke none.
- The year rule is the workhorse. It repaired every case of the 14B writing 2024 into a date the user gave as month and day, including one where the model wrote 2023 and the desk had refused.
- The top_k rule fired on exactly those 5 refusal traps.

**gemma4:e4b gains as much from the check** (76.2% → 90.5%, +14.3, 12 fixed, 0 broken) but stays below 95%. Its remaining misses are refusals and questions on dates it cannot format, which no literal check can repair.

**The keyword parser has little for the check to fix** (90.5% → 92.9%): it never adds a year.

**Post-hoc, outside any verdict and optimistic,** because the rules were designed from these failure classes. Applied to the stored raw outputs of the earlier 14B sealed runs, the check would have taken:
- G10 from 87.9% to 97.8% exact (9 fixed, 0 broken);
- G8b from 92.5% to 97.5% (6 fixed, 0 broken), and it would have refused the one silently clamped "top 8".

**What G10b shows.** Splitting the work this way closes most of the gap to the 30B at the 14B's latency:
- the LLM transcribes;
- a few deterministic rules check each value it wrote against the request text and drop what the text does not support;
- code resolves what remains against the catalogue.

On this fresh set, 14B plus the check reaches 97.6% exact with no silent wrong answers. The 30B reached 97.5% on G8b and 93.9% on the G10 set.

One error class remains, and it is a parsing failure, not a guess: a request that excludes services *and* declines the fault type. It is cheap to target, with a token cap against the hangs and a rule moving a spec field named in the exclusion list back to its field. But both would be another rule chosen from sealed failures, so they would need yet another fresh set to be judged.
<!-- analysis:end -->
