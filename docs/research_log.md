# Research log

**The end goal:** a team of specialist AI models that talk to each other in a compact language they learn themselves. A command centre coordinates them, and a translator lets a person give orders in plain language and read back what happened. The long-term target is a drone swarm: every drone is a specialist (gas sensor, camera, ...) with its own small "brain" running on an ESP32.

**How we test:** every claim is decided by a test written down *before* the run (a "goal", G-number) with a pass bar and a kill bar. Results always come with baselines, several independent training runs and 95% confidence intervals. Data used for a final test ("sealed" sets) is used once. Quick **probes** are exploratory: they show where a real test is worth running.

**Where the evidence points so far:**

- **Learned languages work** in simulation and on sensor data when the channel is tight. They match good hand-designed codes, can be read and written by a monitor, and can be bridged between teams.
- **A learned command centre can follow structured goals**, including combinations it never trained on.
- **The plain-language translator works** (97.6% exact with a code check).
- **Large language models as specialists did not pay off** on real incident data. Small trained specialists sending typed reports did.
- **The drone swarm** needs to talk, but its radio is not the bottleneck. Coordination is, once there are several targets.

## Track: Learned messages between specialists

Can specialists invent their own compact message code, and when does it beat a code a human designed? This is the heart of the end goal.

### R6: learned protocol in the snake squad
- **Date:** 2026-10-23
- **Question:** If a centre and three snake specialists learn their own 3-bit messages from scratch, do they do as well as a hand-designed code?
- **Why:** The first check that learned communication works at all in our lab.
- **What we did:** Trained the centre and the specialists together with evolution strategies (ES), and compared team food against a hand-coded direction message, at observation radius 3 and 5.
- **Result:** On the GPU runs, learned 35.4 against hand-coded 34.9 team food at radius 3, and 27.3 against 27.3 at radius 5 (`learned_protocol_gpu_r5.md`). An earlier 4-run CPU version was worse: learned 22.5 against 30.7, and one run in four formed no code (`learned_protocol.md`).
- **Verdict:** MATCHED
- **What it means:** Learned codes are as good as a good human design, not better, when the channel is roomy enough.
- **Report:** docs/results/learned_protocol_gpu_r3.md

### G2: learned codes where bandwidth binds (plant floor)
- **Date:** 2026-10-24
- **Question:** When each sensor node may send only 1–4 bytes, does a learned code diagnose plant faults better than the best hand-designed code?
- **Why:** The bandwidth-limited edge case (LoRa-class radios, sensor nodes) is where learned compression should matter. It uses real industrial data (the Tennessee Eastman process).
- **What we did:** Four sensor nodes encoded their readings into 1, 2, 4 or 8 bytes with learned codes (end-to-end, k-means, local classifier) or hand designs. A centre diagnosed 17 fault types. 10 replicates.
- **Result:** At 1 byte, the end-to-end and local-classifier codes reach 0.96–0.97 macro-F1 against 0.82 for the best hand design (+0.14); learned k-means reaches 0.83. All 9 learned-vs-hand comparisons are significant.
- **Verdict:** PASS
- **What it means:** The clearest win for learned communication so far. It holds exactly where bytes are scarce.
- **Report:** docs/results/plant_learned_codes.md
- **Chart:** g2_plant

### G2b: do learned codes extend to new faults?
- **Date:** 2026-10-25
- **Question:** Frozen codes trained on 12 faults: can they describe 5 faults never seen, with only 3 labelled examples each?
- **Why:** A real deployment meets faults nobody trained for.
- **What we did:** Trained codes on 12 faults, froze them, then tested on 5 unseen faults across 5 splits × 10 replicates.
- **Result:** No learned code beat the designed fault-ID code (1 byte: 0.832 against 0.837).
- **Verdict:** LEANING KILL
- **What it means:** Learned codes do not generalise to unseen faults better than a designed code.
- **Report:** docs/results/plant_novel_faults.md

### R9: raw hidden states between language models
- **Date:** 2026-10-24
- **Question:** Can two LLM specialists talk by passing internal vectors instead of text?
- **Why:** A "language the models invent" could be their internal states.
- **What we did:** Invoice-like records: model A extracts, model B normalises. Compared JSON text, learned latent vectors and KV-cache handoff.
- **Result:** Latent vectors gave 0% exact. KV handoff is lossless and 43% faster, but about 100,000× bigger than JSON.
- **Verdict:** KILL
- **What it means:** Between LLMs, typed JSON stays the protocol. KV handoff is only a same-machine shortcut.
- **Report:** docs/results/r9_latent_vs_json.md

## Track: A command centre that follows goals

Can a learned centre take a structured order (a "GoalSpec": which snakes eat, how cautiously) and coordinate specialists that never see the order themselves?

### G7: a learned centre follows a GoalSpec
- **Date:** 2026-10-26 to 2026-10-28
- **Question:** Can a learned centre follow goals, including combinations it never trained on, as well as a hand-coded centre?
- **Why:** The command centre is the coordinator of the whole ecosystem. It must obey orders, not one fixed task.
- **What we did:** 21 goals (which of 3 snakes eat × 3 caution levels); trained on 15, tested on 6 held out. Compared (a) a goal-blind centre, (b) goal only at the centre, so it must travel through the 3-bit messages, (c) goal at the centre and the specialists, and (d) a hand-coded centre. 16 training runs each at 10k generations.
- **Result:** On unseen goals, (c) reaches 92% of the hand-coded centre and (b) 85%, against −15% for goal-blind. (b) failed to train in 2 of 16 runs.
- **Verdict:** PASS
- **What it means:** A learned centre can carry orders to specialists through a learned language and generalise to new orders.
- **Report:** docs/results/r7b_goal_conditioned_16runs.md
- **Chart:** g7_curves

### G7c: is that training reliable?
- **Date:** 2026-10-29
- **Question:** Do the "goal only in the messages" runs train reliably, given that 2 of 16 failed in G7?
- **Why:** A method that sometimes silently fails is not a foundation.
- **What we did:** 20 fresh runs, with a pre-registered rule to discard runs whose training-goal score stayed below 12.
- **Result:** No run failed. They reach 95% of the hand-coded centre on unseen goals, level with condition (c). Overall 2 of 36 runs have failed (about 6%).
- **Verdict:** PASS
- **What it means:** Reliable. G7's gap came from two unlucky runs.
- **Report:** docs/results/r7c_reliability.md
- **Chart:** g7_heldout

## Track: Reading and writing the learned language

A language only the machines understand is unsafe. Can a monitor read it, and can an operator write into it?

### G12: a monitor reads and writes the learned language
- **Date:** 2026-10-29
- **Question:** From the 3-bit messages alone, can a decoder tell which snakes were told to eat? Can writing the decoded symbols steer a snake?
- **Why:** The roadmap's "reverse translator" must decode learned payloads for monitoring, and humans need a way to countermand.
- **What we did:** Fitted one small decoder per trained squad on training-goal episodes. It maps symbols to human meanings: eat or do not eat, and where the food is. Tested on unseen goals across 34 squads.
- **Result:** The decoder names the right set of eaters in 94% of episodes. Writing "stop" stops a snake fully; writing "eat" gives 74% of a real eater's food. Caution is not carried. The 34 squads speak 33 distinct dialects; only one pair shares one.
- **Verdict:** PASS
- **What it means:** Learned languages can be monitored and steered, but each team needs its own decoder.
- **Report:** docs/results/r7d_legibility.md
- **Chart:** g12_scatter

### Probe: bridging two teams' dialects
- **Date:** 2026-10-29
- **Question:** Can squad A's centre command squad B's snakes if we translate between their dialects through the shared human meanings?
- **Why:** In an ecosystem, parts trained separately must work together without retraining.
- **What we did:** For 34 squad pairs: A's centre → A's decoder → B's writer → B's snakes, on unseen goals.
- **Result:** Without the bridge: 0% of B's own score. With it: 71%, almost exactly the ceiling of B's writer (73%).
- **Verdict:** PROBE: WORKS
- **What it means:** Separately trained parts can be joined after the fact. The limit is how rich the shared meaning vocabulary is (only 8 coarse situations today).
- **Report:** docs/results/r7e_interlingua.md
- **Chart:** interlingua

## Track: Plain language in, plain language out (the translator)

A person types an order; the translator turns it into a validated GoalSpec. Afterwards a summary is written back in plain language.

### G3: the end-to-end slice
- **Date:** 2026-10-25
- **Question:** Does natural language → translator → GoalSpec → centre → specialists → summary work end to end?
- **Why:** The translator is the human interface of the ecosystem.
- **What we did:** 174 hand-written requests, never used for tuning (a sealed set), through qwen2.5-coder:14b with a parse-then-validate policy.
- **Result:** 95.9% exact. The task score from plain language is within the confidence interval of the hand-written-spec run. Round trip 100%; p95 latency 1.4 s.
- **Verdict:** PASS
- **What it means:** A local 14B model is a dependable translator.
- **Report:** docs/results/r7a_step2.md

### G3b: can a small distilled model replace the 14B?
- **Date:** 2026-10-25 to 2026-10-29
- **Question:** Can a small fine-tuned model (1.7B, then 4B) translate as well, so it fits beside other models?
- **Why:** A 9 GB translator crowds a 12 GB GPU.
- **What we did:** Fine-tuned LoRA students on templates plus thousands of verified paraphrases, chosen on a dev set, then one sealed run each.
- **Result:** 1.7B: 82.8% (twice; this report). 4B: 84.1% (`docs/results/r7a_step4_q4b_sealed.json`).
- **Verdict:** KILL
- **What it means:** A bigger student is not the fix. The gap is the shift from template-like training phrasing to real phrasing. The prompted 14B stays.
- **Report:** docs/results/r7a_step3.md

### G8/G8b: the whole chain on real incidents
- **Date:** 2026-10-27
- **Question:** Can one translator route requests across domains (squad, plant, incidents) and drive the real incident team?
- **Why:** Tests the ecosystem as a product on real data.
- **What we did:** Requests like "what broke on the 17th?" were resolved against a catalogue of real incidents, run through the trained team, then summarised.
- **Result:** G8 failed on dates without a month. After fixing the spec (G8b): 92.5% exact, routing 100%, summaries keep 100% of facts.
- **Verdict:** PASS (G8b)
- **What it means:** The plain-language desk works on real incidents once the spec handles partial dates.
- **Report:** docs/results/r11b_ecosystem.md

### G10/G10b: growing the desk and checking the translator
- **Date:** 2026-10-28
- **Question:** Does the desk extend to a new system without prompt changes? Can a few code rules catch the translator's guesses?
- **Why:** Adding a system should not mean rewriting prompts, and silent wrong answers are the worst failure.
- **What we did:** Added Train Ticket and 62 new incidents (G10, `r12_growth.md`). Then added a "literal check" that removes any year, month or count the request did not actually state.
- **Result:** Exact match rose from 84.5% to 97.6% with the check, with zero silent wrong answers. One error class remains, so not every criterion passed.
- **Verdict:** NOT PASSED (3 of 4)
- **What it means:** Small code checks around an LLM fix most of its mistakes. A 30B model was not better (92.9%).
- **Report:** docs/results/r12c_literal_check.md

## Track: Specialists on real data (incident triage)

Real microservice incidents (the RCAEval benchmark): find the broken service and the fault type. This is the real-data test of specialists plus a centre.

### G4, G4b, G4d, G4e: LLMs as specialists
- **Date:** 2026-10-25 to 2026-10-27
- **Question:** Do teams of LLM specialists, each reading part of the data, beat one model or a simple statistical method?
- **Why:** The original plan used LLM specialists on a shared base model.
- **What we did:** Four designs, run on 62–120 test incidents and compared with BARO (a statistical anomaly ranker) and simple code:
  - LLM log readers (`r10_log_triage.md`);
  - hybrid teams with LLM centres (`r10_step2.md`);
  - LLM teams on code-level faults (`r10_step4.md`);
  - an LLM picking from a shortlist (`r10_step5.md`).
- **Result:** Every LLM team lost to simple code. For example, the typed LLM team got 4% top-1 against BARO's 88%, and an LLM picking from a shortlist made it worse (77% against 81%).
- **Verdict:** KILL (4 tests)
- **What it means:** On this data, LLMs are not good specialists. Code and small trained models are.
- **Report:** docs/results/r10_log_triage.md

### G4c: a small trained fault specialist
- **Date:** 2026-10-27
- **Question:** Does a small trained classifier that sends a typed report (six fault probabilities) make the team better?
- **Why:** After the LLM failures: train a tiny specialist for exactly the part that was weak, the fault type.
- **What we did:** A 59-feature softmax classifier trained on other systems' incidents. The rule centre takes the fault type from its report.
- **Result:** Joint accuracy 68.3% against BARO's 45.0% (+23 points).
- **Verdict:** PASS
- **What it means:** Small trained specialists with typed messages are the working recipe on real data.
- **Report:** docs/results/r10_step3.md

### G11a/G11b: does it replicate, and can the centre be learned?
- **Date:** 2026-10-28
- **Question:** Does G4c's gain hold on 375 fresh incidents? Does a learned centre beat the simple rule centre?
- **Why:** One result on one dataset is not enough, and the centre is the next thing to learn.
- **What we did:** Ran the frozen team on RE1 (new data), and trained a ranking centre on RE1 then tested it on RE2.
- **Result:** +9.3 points pooled, but not on Online Boutique (missing metric kinds). The learned centre gained only +2.4, not significant.
- **Verdict:** PARTIAL / KILL
- **What it means:** The specialist mostly transfers. Learned centres need far more data than real incidents give.
- **Report:** docs/results/r13_real_centre.md

### Probe: how many labels does a new system need?
- **Date:** 2026-10-29
- **Question:** If a new deployment labels a few of its own incidents, does the specialist recover where it failed?
- **Why:** Real deployments always differ from the training data.
- **What we did:** A small adapter on top of the frozen specialist, fitted on k = 5–40 labelled incidents of the target system. Checked with random, by-repetition and unseen-service splits.
- **Result:** Online Boutique goes from 52% to 83–86% joint with 20 labels, and holds for unseen services (54% → 78.5%). A model trained from scratch on the same labels does as well.
- **Verdict:** PROBE: PRACTICAL
- **What it means:** About 20 labels per new system fix transfer. The feature design carries over, not the trained weights.
- **Report:** docs/results/r13b_kshot.md
- **Chart:** kshot

## Track: Growth (adding specialists)

Can a new specialist join a working team without retraining everyone?

### G1: a newcomer joins the snake squad
- **Date:** 2026-10-24
- **Question:** Can a new specialist join and reach 90% of a jointly trained one at ≤ 30% of the compute, with the old team unchanged?
- **Why:** An ecosystem must grow one specialist at a time.
- **What we did:** Added a newcomer either copied from an existing specialist or trained from scratch, under learned and hand-coded protocols.
- **Result:** A copied newcomer joins at almost zero cost; a from-scratch one is costly under both protocols.
- **Verdict:** PASS (copy) / FAIL (scratch)
- **What it means:** New specialists should start as copies of existing ones (LoRA from LoRA).
- **Report:** docs/results/add_specialist.md

### G9/G9b: a newcomer joins the incident team
- **Date:** 2026-10-28
- **Question:** Can a log-reading newcomer join the real incident team, gaining on code faults without hurting the old ones?
- **Why:** Growth on real data, not just simulation.
- **What we did:** Added a log-novelty newcomer through the same command centre, keeping the old team frozen.
- **Result:** +22.6 points on code-level faults, but −3.3 on the old resource faults, in two tries.
- **Verdict:** KILL
- **What it means:** A rule centre cannot tell when to trust the newcomer. That needs a signal the reports do not carry yet.
- **Report:** docs/results/r12_growth.md

## Track: Drone swarm (open direction)

The user's long-term target: heterogeneous drones, each with two ESP32s, a gyro and accelerometer, two positioning sensors and 2–3 accessories, each with its own specialist brain.

### G13: does the swarm need to talk, and does the radio bind?
- **Date:** 2026-10-29
- **Question:** In a simulated gas-leak search, do drones need to talk? Do tiny 1–2 byte messages lose much?
- **Why:** Learned compression only matters if communication matters *and* bandwidth is tight.
- **What we did:** 3 gas-sensing and 3 camera drones in a 30 m arena, with drifting positions and a lossy radio. Hand-designed silent and talking swarms at 1–8 bytes per message, 3,000 episodes per setting.
- **Result:** Talking adds +24 points (73% → 98% found). But 2 bytes are as good as 8, and 1 byte keeps 89% of the gain.
- **Verdict:** RADIO DOES NOT BIND
- **What it means:** The swarm must talk, but compression is not where learning helps here.
- **Report:** docs/results/r14_swarm_talk.md
- **Chart:** g13_swarm

### Probe: several gas leaks at once
- **Date:** 2026-10-29
- **Question:** With 3 leaks, how much better could the swarm do with ideal coordination than with a tuned hand design?
- **Why:** If hand design leaves little room, learned coordination has nothing to win.
- **What we did:** Tuned a hand-designed gossip swarm. Compared it with a bound where a leak's exact position is known to all at the first sniff, cameras are split ideally and gas drones move on.
- **Result:** All three found: hand design 61.9%, ideal coordination 93.1% (+31 points, 77 steps faster). With one leak the gap is only 2 points.
- **Verdict:** PROBE: ROOM FOR LEARNING
- **What it means:** The swarm's open problem is coordination (who goes where, when to move on). That is the job of learned per-drone brains. Next: train them.
- **Report:** docs/results/r14_multi_probe.md
- **Chart:** multi_probe

### R14 step 2: learned per-role brains for the swarm
- **Date:** 2026-10-29
- **Question:** Can small learned "brains", one for gas drones and one for camera drones, coordinate 3-leak searches better than the tuned hand design?
- **Why:** The probe left a 31-point gap that only better coordination can close. These brains are the first step towards a specialised brain per drone type.
- **What we did:** Kept the hand design's radio and memory; each drone scores its candidate destinations (explore, a known hotspot, its own best reading) with its role's network (about 270 weights each). Trained with evolution strategies on 3 leaks (16 runs × 1,500 generations, 7.3 h each), tested on the probe's 3,000 episodes with 1–4 leaks. Good sign, fixed in advance: all three found at least 10 points more often than the hand design, CI above 0.
- **Result:** All three leaks found in 78.3% of episodes, against 62.5% for the hand design and 93.2% for the coordination bound: +15.9 points (± 0.5 over the 16 runs; even the worst run reaches 76.1%). The brains close half the gap. With 4 leaks, which they never trained on, the gain grows to +24.4 points (58.0% vs 33.6%); with 2 leaks +5.9; with 1 leak they match the hand design (97.0% vs 96.6%). Most runs reached their best validation score within a few hundred generations.
- **Verdict:** GOOD SIGN
- **What it means:** Learned per-role coordination beats a tuned hand design by a wide margin, and the gain grows as the mission gets harder. The brains are tiny (two 15-16-1 networks), so they would run on an ESP32-S3 with room to spare. Before this counts for hardware, step 3 tests whether the gain survives worse radio, drift, weaker sensors and drones going down, and whether training on randomised conditions makes it hold.
- **Report:** docs/results/r14_brains.md
- **Chart:** r14_brains

### R14 step 3: do the brains survive worse conditions?
- **Date:** 2026-10-29
- **Question:** Does the learned brains' gain hold when the radio, sensors or drones get worse than in training? Does training on randomised conditions make it hold better?
- **Why:** Real flights won't match the one training setting. A gain that vanishes with more radio loss or a lost drone would not be worth flashing onto hardware.
- **What we did:** Played the step-2 brains, the hand design and the sensed oracle in 10 harder conditions, retuning neither side: 50% and 70% radio loss, 3× position drift, a weak gas sensor, narrow and wide plumes, 1 or 2 drones going down mid-mission (new in the simulator), only 4 drones, and a 40 m arena. Each cell has 3,000 episodes at 3 leaks. A second set of 16 brains trains on a random condition every generation, with the same budget. Bars fixed in advance: (a) the step-2 brains beat the hand design in every condition, CI above 0; (b) the randomised brains beat the step-2 brains on average over the harder conditions, and lose at most 3 points at nominal.
- **Result:** (a) passes: the brains beat the hand design in all 10 conditions. Gains: +17.0 points at 70% radio loss, +21.6 at 3× drift, +24.2 with a weak gas sensor, +21.0 with 2 drones down, +19.6 with only 4 drones and +15.3 in the 40 m arena. The smallest is +5.4, with a narrow plume. With a wide plume the hand design collapses to 10.4% while the brains reach 67.5%. The sensed oracle's fixed smell threshold and move-on rule fail there too (46.5%), so it is a bound only near the settings it was built for; it also does not know which drones went down. (b) passes its bar, but the effect is tiny: the brains trained on randomised conditions are +0.6 ± 0.2 points better averaged over the harder conditions and equal at nominal. Most of that is the wide plume (+4.6, 72.1% vs 67.5%). Outside the randomised range they are slightly worse (4 drones −0.6, 40 m arena −0.9). The bar set no minimum size, so it is met with almost nothing to spare in practical terms.
- **Verdict:** ROBUST; randomised training adds little
- **What it means:** The learned coordination is not tuned to one setting. Its gain grows when sensing or positions get worse, and it survives losing a third of the swarm. The radio barely matters: the brains lose only 1 point going from 30% to 70% loss. This result justifies asking for desk hardware. Randomised training costs nothing at nominal and helps a little where the plume is unusual, so flight brains can use it, but the brains trained at one setting were already robust.
- **Report:** docs/results/r14_robust.md
- **Chart:** r14_robust

### R14 step 4a: the swarm brains in int8, ready for the ESP32
- **Date:** 2026-10-29
- **Question:** Do the learned swarm brains keep their gain when they run in 8-bit integers, as they would on an ESP32-S3?
- **Why:** Microcontrollers run integer maths much faster than floats, and their ML libraries (ESP-DL, TFLite Micro) expect int8 networks. The bench step needs brains that run that way.
- **What we did:** Ran both sets of 16 trained brains (step 2 and randomised) the way the chip would: int8 weights and inputs, 32-bit integer sums and a 256-entry tanh table. Played them on the same 3,000 test episodes as the float versions, at nominal, at 70% radio loss and with 2 drones down. Bar fixed in advance: at nominal, int8 within 1 point of float and still above the hand design. The best-validation randomised brain (run 108) was exported as a C header.
- **Result:** No measurable loss. At nominal, int8 minus float is −0.1 ± 0.6 points (step 2) and +0.3 ± 0.6 (randomised). Every condition stays within ±0.3. The header holds both brains in 4 KB: two 15-16-1 networks, about 270 weights each.
- **Verdict:** PASS
- **What it means:** The brains are ready for the chip as they are, with no retraining for quantisation. The bench step can start from `hardware/esp32/swarm_brains_int8.h` and time the brains on a real ESP32-S3.
- **Report:** docs/results/r14_int8.md

## Track: OpenSwarm brains (the chosen direction, 2026-10-29)

The user's drone platform, OpenSwarm (`D:\Projects\drone`), becomes the embodiment of the ecosystem. Its hardware: an ESP32-S3 flight controller, a K210 vision module, UWB ranging, optical flow and GPS. Its protocol: a LEADER sends `BEHAVIOR_CMD`s and SCOUTs send `DETECTION` packets over ESP-NOW. This track builds the specialist brains it needs and tests them in simulation before hardware.

**Why this direction:**
- It is where the user wants the work to go, with real hardware and a deadline (31 Dec 2026).
- Its design matches what held up here: typed messages, a coordinating leader, specialist roles and a plain-language front end.
- Dropped: incident triage (no real data), more snake-squad work (done its job), learned compression (ESP-NOW has room) and LLM specialists (lost to small trained models).

### R15: the scout brain (person detector for the K210)
- **Date:** 2026-10-29
- **Question:** How well can a person detector small enough for OpenSwarm's K210 module find people, and how does accuracy trade against size and compute?
- **Why:** The SCOUT role's DETECTION packet needs a person's centre and confidence. That brain must fit the K210 (a few MB of memory, int8). OpenSwarm's plan names YOLOv2-tiny at 224×224 but has no trained model yet.
- **What we did:** COCO 2017 persons, letterboxed to 160×160. A MobileNet-style CenterNet detector at widths 0.25, 0.5 and 0.75, using only K210-friendly layers. Trained on train2017 (118k images, 40 epochs) on the GPU and scored on the 5,000 val2017 images: AP50, AP50 for persons at least 32 px tall, and person-present AUC. Exported to ONNX for the K210 toolchain. An exploratory measurement: no bar was set in advance.
- **Result:** Smallest (115 KB int8, 20 M multiply-adds per frame): AP50 0.24, tall persons 0.48, present AUC 0.91. Largest (959 KB, 159 M): AP50 0.30, tall persons 0.58, present AUC 0.93. Width 0.5 (434 KB) sits between. All three fit the K210 and export to ONNX.
- **Verdict:** MEASURED: FITS THE K210
- **What it means:** A sub-megabyte brain reliably tells whether a person is in view and finds people who fill part of the frame. Small, distant people are the weak spot, and an 8× bigger model buys only 6 points there. Drones see exactly those people, from above. Next: test it on aerial footage (VisDrone) and fine-tune it for the drone's view.
- **Report:** docs/results/r15_scout_brain.md
- **Chart:** r15_scout

### R15b: the scout brain from a drone's view
- **Date:** 2026-10-29 (running)
- **Question:** Does the COCO-trained scout brain find people in drone footage, and how much does training on aerial images help?
- **Why:** COCO photos are taken from the ground, while a SCOUT drone looks down from altitude, where people are a few pixels wide and seen from above. R15 showed that small people are the brain's weak spot.
- **What we did:** VisDrone2019-DET (drone footage from 14 cities; 6,471 training frames and 1,610 test frames, 16–25 people per frame). Every frame is cut into 224×224 tiles (OpenSwarm's input size) at native and half resolution. Three brains at widths 0.25, 0.5 and 0.75: R15's COCO models unchanged (zero-shot), the same models fine-tuned on aerial tiles, and the same training from random weights (scratch). Scored once on the test frames. Besides AP50, "centre AP" counts a detection when its centre lands on the person, because that is what the DETECTION packet carries. Scores are also broken down by person size. Bars fixed in advance, at width 0.5: (1) fine-tuning beats zero-shot by ≥ 10 AP50 points, and (2) starting from COCO beats scratch, both with paired intervals above 0 for every seed.
- **Result:** Running (about 12 hours on the GPU).
- **Verdict:** Running
- **What it means:** Pending.
- **Report:** docs/directions/drone_swarm.md

### R17: a leader that plans the person search
- **Date:** 2026-10-29
- **Question:** Does a leader that reads the mission (area, number of drones, how good the scout's detector is) plan OpenSwarm's person search better than one well-tuned plan used for every mission?
- **Why:** This is ladder step 3, a learned centre that follows the mission. It also answers a practical question for the hardware: what altitude, threshold and confirmation scheme to fly. A 5-minute battery forces a trade: higher sweeps cover more ground but shrink people below what the K210 detector can see, and false alarms send the team to the wrong place unless a drone confirms.
- **What we did:** A new numba simulator of one sortie: scouts fly lanes, confirmers fly to detections and look again, every drone gets one battery, and a detection curve includes false alarms. Missions are random (60–240 m sides, 3–8 drones, detectors from good to poor). We compare five systems on 40 held-out missions × 2,000 episodes: the OpenSwarm default (sweep at 5 m, the first detection ≥ 0.7 is the find), a hand two-tier plan, one plan ES-tuned for all missions, an ES-trained planner that reads the mission, and per-mission tuning (the room). Bars fixed in advance: per-mission tuning beats the one plan by ≥ 5 points (room), and the planner beats it by ≥ 3 points with CI above 0 (good sign).
- **Result:** The planner that reads the mission scores 64.6 points against 46.6 for the one tuned plan: +18.0 ± 6.1 (bar ≥ +3: met). Per-mission tuning reaches 69.1, so the room was +22.6 (bar ≥ +5: met) and the planner captures about 80% of it. All 16 planner runs (63–66) beat all 16 one-plan runs (46–47); the planner wins on 35 of 40 missions, and its worst loss is 4 points. It finds the person in 69% of searches with 2.2% wrong finds, against 54% and 3.6% for the one plan. The OpenSwarm default (first detection ≥ 0.7 is the find) is misled by false alarms: 52% wrong finds. That figure depends on the assumed false-alarm rates. The plans make sense: the planner sweeps between 3 and 17 m (averaged over runs), higher over larger areas and with better detectors, where the one plan always sweeps at 4.6 m; with many false alarms it confirms with more looks at a higher threshold.
- **Verdict:** GOOD SIGN
- **What it means:** The leader should plan from the mission, not run one fixed search. The area and the detector both set the best altitude, so the camera choice matters as well as the plan: a detector that sees people at fewer pixels lets the scouts fly higher and cover more ground on one battery. Next (R17 step 2): R15b's measured detection curve as a held-out detector, then candidate camera modules compared by search score, now that the hardware is ours to choose. Before that, a weight and power cost per module.
- **Report:** docs/results/r17_mission.md
- **Chart:** r17_mission

### R16 (G14): a plain-language ground station for OpenSwarm
- **Date:** 2026-10-29
- **Question:** Can an operator command the swarm in plain words ("sweep the parking lot at 10 m") and have the ground station send exactly that command, ask when unclear and refuse what the swarm can't do?
- **Why:** OpenSwarm's leader sends one of 8 behaviours (hover, formation, boids, search grid, perimeter, person-follow, light show, return home) in a 64-byte BEHAVIOR_CMD. A wrong command is a swarm flying somewhere nobody asked, so the key number is how often a request ends as the wrong command.
- **What we did:** A local model only extracts the behaviour and its stated values. Code does the rest: it fills OpenSwarm's documented defaults, resolves place names from a site map ("the old barn" becomes coordinates), validates against a schema in which each behaviour takes only its own parameters, checks the swarm state (vision scout present, indoor 3 m ceiling, battery, enough drones, no-fly places) and encodes the packet. The largest packet uses 35 of the 64 bytes. A separate agent wrote 150 sealed test requests and 60 dev requests from a labelling guide, without seeing the prompt. Bars fixed in advance, for qwen2.5-coder:14b: exact on ≥ 90% of commands, wrong commands on ≤ 3% of requests, and ≥ 20 points above a keyword parser. Smaller models (gemma4 e4b and e2b) test whether a laptop could run the station.
- **Result:** On the 150 sealed requests qwen2.5-coder:14b got 93.3% of commands exact (bar ≥ 90%: met) and beat the keyword parser by 54 points (bar ≥ 20: met). It sent a wrong command on 5 requests, 3.3% (bar ≤ 3% with upper bound ≤ 6%; the upper bound is 8%): missed. Of the five: (1) a station bug: the number parser reads "two and a half metres" as 2, so it rejected the model's correct 2.5 and the retry flew the default spacing; (2) "formation at 7 m" became 7 m spacing instead of altitude; (3)–(5) the model guessed a command for two vague requests ("do the same as before", "go to the usual spot and do your job") and for a two-step one ("hover and rth afterwards"). Three of these came out of a retry that stripped invented values and left a bare behaviour with defaults. Smaller models trade helpfulness for safety: gemma4 e2b 76.7% exact with 1.3% wrong, e4b 48.9% exact with 0 wrong. The keyword parser: 38.9% exact, 14% wrong. The dev set needed four versions: v2 fixed questions written in the refusal field, v3 a schema-order bug (altitude written into the wrong field), v4 numbers placed in the wrong setting.
- **Verdict:** NOT PASSED (narrowly)
- **What it means:** The model reads plain commands well; the failures are in the edges the station's code should own. One is a plain bug (spoken fractions). One is a convention to settle ("at 7 m" means altitude). The rest are a rule to add: when a retry strips everything but the behaviour from a request that never named one, ask instead of sending defaults. Wrong commands stay far below the 10% kill line, so the direction lives. Next (G14b): those fixes on dev, then a fresh sealed set written the same way, because this one is now spent for this system.
- **Report:** docs/results/r16_ground_station.md
- **Chart:** r16_ground
### R17 step 1b: does the search planner survive a worse world?
- **Date:** 2026-10-29
- **Question:** Does the mission-reading planner keep its lead when the person walks, the GPS is worse, there is a headwind, the battery is cold, the area is larger or there are more drones?
- **Why:** The planners trained on a still person, 1 m GPS error, 4 m/s drones and a 300 s battery. Real searches won't match that.
- **What we did:** Replayed the 16 mission-reading planners, the 16 one-plan runs and the hand plans, none retrained, on the 40 held-out missions in 7 worse conditions (2,000 episodes per cell). The bar was fixed in advance with a minimum effect size: the planner beats the one plan by ≥ 3 points, CI above 0, in every condition.
- **Result:** FAIL in 2 of 7 conditions. The planner keeps a clear lead with a headwind (+17.4), a cold battery (+17.0), areas 1.6× larger (+14.0) and 10 drones (+13.6), the last two outside its training range. It falls short with a person walking 1.2 m/s (+3.1 ± 4.2) and with 3 m GPS error (+2.2 ± 3.9). The bigger finding: a walking person breaks every plan that confirms. Even at 0.5 m/s finds drop from 69% to 6% (one plan 54% → 3%), while OpenSwarm's default, which never confirms, still finds 29%. A confirmer flies to where the person was 30–60 s earlier and hovers over a few square metres. With 3 m GPS error part of the drop is the scoring: a find must land within 5 m, and the GPS error alone puts about a quarter of correct detections outside that.
- **Verdict:** NOT ROBUST
- **What it means:** Planning from the mission is worth it, but the search tactic it plans is brittle: confirming by hovering over the reported spot only works for someone who stays put. The fix is the tactic, not the planner. The scout that sees a person should keep it in view (OpenSwarm already has a person-follow behaviour) and hand it to the confirmer, which should search around the latest sighting. Next (R17 step 2): add that hand-off, train on missions that include walking people and worse GPS (R14 showed randomised training costs nothing), and score finds within a radius that grows with the GPS error.
- **Report:** docs/results/r17_robust.md
- **Chart:** r17_robust

### R17 step 2: a hand-off for people who walk
- **Date:** 2026-10-30
- **Question:** If the scout that sees a person keeps it in view until a confirmer takes over, does the search survive a walking person? Does a planner trained on harder missions keep its lead?
- **Why:** Step 1b showed every confirming plan collapses when the person walks, because confirmers hover over where the person was.
- **What we did:** A new plan setting, the hand-off: the detecting scout holds the sighting for up to N seconds, as OpenSwarm's person-follow does, and the candidate moves with each re-sighting. Harder missions: half have a walking person (0.2–1.5 m/s); all have 1–4 m GPS error, wind, a 220–320 s battery, and a find radius that grows with the GPS error. The leader reads GPS quality, wind and battery, never whether the person walks. 16 one-plan and 16 mission-reading runs, tested on 40 held-out missions. Bars fixed in advance: (a) the hand-off adds ≥ 10 points on walking missions; (b) the planner beats the one plan by ≥ 3; (c) it beats the step-1 planner by ≥ 10 on walking missions; (d) it stays within 3 points of the step-1 planner when the person stands still. A pipeline smoke test previewed bar (a), which involves no training, before the bars were committed; they were not changed.
- **Result:** NOT YET: 2 of 4 bars met. (b) Reading the mission still pays: the new planner beats the one plan by +11.4 ± 3.3 points. (c) The walking-person collapse is fixed: on walking missions the new planner scores 29.3 against 2.7 for the step-1 planner (+26.6 ± 10.9). (a) A fixed 60 s hand-off added to the hand plan gains +9.2 ± 5.1 on walking missions, just under the +10 bar, and costs 6 points when the person stands still. (d) Failed: on step 1's still-person missions the new planner scores 49.2 against 64.6 (−15.3 ± 3.6). It cannot tell whether the person walks, so it hedges on every mission. Tuning each mission alone reaches 38.7 overall (34.6 walking, 44.1 still), so the planner captures about half of the room above the one plan (17.9).
- **Verdict:** NOT YET
- **What it means:** The hand-off is the right tactic for someone who walks, and a planner that reads the mission still beats one fixed plan. But a single plan has to hedge between a still and a walking person, and that costs 15 points when the person is still. That is missing information, not a missing tactic: an operator usually knows whether the person is injured or moving. (Correction, 2026-10-30: step 3 showed this explanation is wrong, because a correct moving hint does not recover the loss.) Next: make "may be moving" a mission input (the ground station can ask for it), and let the leader switch to the hand-off when one person keeps being re-sighted in different places.
- **Report:** docs/results/r17b_handoff.md
- **Chart:** r17b_handoff

### R17 step 3: tell the leader whether the person may be moving
- **Date:** 2026-10-30
- **Question:** If the operator says whether the person may be moving, does the planner stop hedging, keeping step 2's gain on walking people without losing step 1's score on still ones?
- **Why:** Step 2's planner could not tell a still person from a walking one and gave up 15 points on still-person missions. An operator usually knows ("injured hiker" vs "lost child"), and the ground station can ask.
- **What we did:** Version 3 planners read one more input: the operator's hint (+1 may be moving, −1 still), right 85% of the time in training. 16 runs with step 2's budget. Tested on the same held-out missions with the hint 85% right, always right, always wrong and absent, and on step 1's still-person missions. Bars fixed in advance: (a) on still-person missions at most 3 points below the step-1 planner; (b) +3 over the step-2 planner with the 85% hint, CI above 0; (c) at most 3 points below the step-2 planner on walking missions.
- **Result:** NOT YET: 1 of 3 bars met. The planner uses the hint (told "still" it holds a sighting 26 s on average, told "may be moving" 95 s), but the hint barely moves the score. With the hint always right it scores 29.4 against step 2's 29.3; with the 85% hint 28.4 (−0.9 ± 1.8; bar +3). A wrong hint costs 7 points (22.3). Walking people are kept (−2.3, bar ≥ −3: met). On step 1's still-person missions it stays 15.6 ± 2.9 below the step-1 planner (bar ≥ −3: missed). A follow-up check on 20 of those missions (300 episodes each): removing the hand-off from its still-person plans does not close the gap (41.5 → 39.3 against 60.4). The difference lies elsewhere: it sweeps higher (10.6 m vs 6.8 m) and keeps more drones back to confirm (39% vs 24%).
- **Verdict:** NOT YET
- **What it means:** Step 2's explanation was wrong: the loss on still people was not hedging over a possibly walking person, since a correct hint does not recover it. One small network trained over the broader, harsher missions plans worse for the nominal case even when told the person is still, while per-mission tuning finds better plans (44 vs 30 on still people). That points to a training limit, not missing information. A larger planner (32 hidden units, 6,000 generations) is now tested against the same bars. If it also fails, the leader should keep separate planners and let the hint pick one.
- **Report:** docs/results/r17c_hint.md

### G14b: the ground station, fixed, on a fresh sealed set
- **Date:** 2026-10-29
- **Question:** With the three problems G14's misses exposed fixed, does the station meet the same bars on requests it has never seen?
- **Why:** G14 missed its safety bar by one wrong command in 150. Its sealed set is now spent: we read its misses to find the fixes, so it cannot also judge them.
- **What we did:** v5 of the station reads fractions and decimals in words ("two and a half"), treats "at N m" as an altitude unless another word names the setting, and asks instead of sending a bare behaviour with defaults after a retry. On the stored dev replies: qwen 0 wrong commands in 60, no right answer lost. A separate agent wrote 150 fresh sealed requests from the labelling guide alone, with those conventions written down. Ten of its texts happened to match earlier requests exactly and were rewritten before the set was pinned. Same bars as G14.
- **Result:** NOT PASSED again, narrowly on two bars. qwen 14B: 88.9% exact (bar ≥ 90%), 4 wrong commands in 150 = 2.7% (the rate meets ≤ 3% but the upper bound is 6.6%, over 6%), +46 over keywords (met). The fixes did their job: the "at N m" rule caught three commands the model would have flown with the altitude in the wrong setting (they became questions, which costs exactness, not safety). The four wrong commands: a two-step request ("search the orchard then return home" flew the search), a vague one ("get ready for the demo" hovered), a landing request flown as return-home, and a bare "half" the number parser does not read. One prompt gap: the model only sees canonical place names, so it asked about "motorway" and "forest" instead of mapping them to the site map's aliases. gemma4 e2b came close on the same set: 85.6% exact, 2.7% wrong; e4b 60.0% exact, 0.7% wrong.
- **Verdict:** NOT PASSED (narrowly)
- **What it means:** Two sealed rounds land within a hair of the bars, and each round's fixes work but uncover the next edge. The remaining wrong commands are mostly the model guessing at vague or two-step requests. A real station would show the operator the parsed command and send it on one tap; with that confirm step the share of wrong commands that reach the drones drops to whatever the operator overlooks. Rather than a third round of patches, the next ground-station step should test that confirm-before-send flow and give the model the site map's aliases, with a fresh sealed set.
- **Report:** docs/results/r16b_ground_station.md
- **Chart:** r16b_ground

