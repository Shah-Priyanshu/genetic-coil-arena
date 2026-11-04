# Open direction: a drone swarm of heterogeneous specialists

Status: **open option, being explored** (2026-10-29). Not adopted. It becomes a track only if the
simulated tests below show good signs.

## The sketch

Each drone carries:

- two ESP32s:
  - a **flight controller** running stabilisation from the gyro and accelerometer, with 4 motors;
  - a **companion** running the specialist policy, the radio and the sensors;
- two positioning sensors:
  - an **environment** sensor for position in the world: optical flow plus a height ToF sensor (as on a flow deck), or UWB anchors, or GPS outdoors;
  - a **swarm** sensor for position relative to the other drones: UWB two-way ranging, or ESP32 Wi-Fi FTM;
- 2–3 **accessories** that make it a specialist: gas sensor, camera, thermal array, microphone, a long-range radio, and so on.

## How it maps onto this project

| Ecosystem piece | In the swarm |
|---|---|
| Specialists | Drones; the accessories give each one a role, so the team is heterogeneous by construction |
| Learned communication | Compact messages between companion ESP32s, over ESP-NOW, piggybacked on UWB ranging frames, or over LoRa |
| Command centre | A ground station or leader drone that turns a mission GoalSpec into messages (G7: a learned centre follows a GoalSpec, including held-out goals) |
| Translator | Natural language to a `swarm.mission` GoalSpec on the ground station (G3, G10b: qwen2.5-coder:14b with the literal check) |
| Monitor | Reading and writing the learned messages (G12: per-squad decoders read 94% and steer with efficacy 0.74) |
| Growth | A drone with a new accessory joins by copying a policy (G1) |
| Two layers | Flight ESP = the fast loop; companion ESP = the specialist; the ground station runs the object layer |

The models fit easily. G7's squad networks have about 5,600 parameters. That is about 5,600 multiply-adds per decision, which is negligible for a 240 MHz ESP32-S3 at 2–50 Hz (an estimate, not a measurement; ESP-DL or TFLite Micro with ESP-NN would run them in int8). LLMs stay on the ground.

## Hardware facts that shape the tests

- **Radio budget.**
  - ESP-NOW carries 250-byte packets (v1); ESP-IDF 5.4+ also speaks v2 with up to 1,470 bytes ([ESP-IDF docs](https://docs.espressif.com/projects/esp-idf/en/latest/api-reference/network/esp_now.html)).
  - With ESP-NOW and a swarm of about 10, bytes do not bind. They bind when messages ride on UWB ranging frames (a few bytes per exchange), over LoRa (tens of bytes per second under duty-cycle rules), or in large swarms sharing one channel.
- **Swarm positioning.**
  - UWB (DW1000/DW3000) gives about 10 cm in good anchor setups ([arXiv 2403.10194](https://arxiv.org/abs/2403.10194v1)).
  - ESP32 Wi-Fi FTM is supported on the ESP32-S2/S3/C3/C6 at no extra hardware cost, but gives about 1–3 m indoors ([FTM survey](https://arxiv.org/pdf/2509.03901); [ESP32-S3 channel-diversity study](https://www.tkn.tu-berlin.de/bib/roesler2025improving/)).
- **Environment positioning.** A PMW3901 flow sensor plus a VL53L1x ToF sensor (Crazyflie Flow deck v2; the ESP32-S3 LiteWing drone) gives position hold by dead reckoning, so drift grows with time ([Flow deck datasheet](https://cdn.robotshop.com/media/B/BCZ/RB-Bcz-69/pdf/flow_deck_2-datasheet.pdf), [LiteWing](https://circuitdigest.com/litewing)).
- **Flight stack.** Espressif's ESP-Drone (a Crazyflie firmware port, ESP32-S2/S3) flies stabilise, height-hold and position-hold modes. It has had limited support since December 2022 ([ESP-Drone](https://docs.espressif.com/projects/espressif-esp-drone/)).
- **Closest prior work.**
  - *Sniffy Bug* (TU Delft, 2021): gas-seeking nano quadcopters with UWB relative localisation that share gas readings over UWB. The parameters of a hand-designed state machine were evolved in simulation and beat manual tuning in real flights ([arXiv 2107.05490](https://arxiv.org/abs/2107.05490v1)).
  - *Swarm gradient bug algorithm* (Science Robotics, 2019): 6 Crazyflies explored about 80% of an office in 6 minutes ([Bitcraze](https://www.bitcraze.io/2019/10/enabling-swarm-exploration/)).
  - What this project would add: **learned** messages and a **learned, goal-following** centre, typed missions from natural language, and a monitor that reads and writes the learned language.

## Test ladder (each step must show a good sign before the next)

1. **Does the swarm need to talk, and does its radio bind?** (R14 step 1, goal G13; simulation, CPU, minutes.)
   - Hand-designed silent and talking swarms on a heterogeneous gas-source search, at 1–8 bytes per packet, with packet loss and positioning drift.
   - Good sign: talking adds at least 15 points of success, **and** 1 byte keeps at most two thirds of that.
2. **Learned messages where the radio binds** (simulation, CPU-heavy ES runs).
   - The G2 analogue: learned codes against the best hand code at 1–2 bytes, with loss and drift.
   - It needs a task where step 1 found the radio binding. Candidates: several sources, a large swarm on one channel, or messages piggybacked on UWB ranging.
3. **A learned centre follows a `swarm.mission` GoalSpec** (simulation).
   - The GoalSpec covers target type, area, which roles, time limit and risk.
   - Held-out goals as in G7; the translator from G10b in front; the G12 monitor behind.
4. **Bench, no flight.**
   - 3 ESP32-S3 boards on a desk: measure ESP-NOW latency and loss and FTM (or UWB) ranging error.
   - Run the trained policy networks on the chips and time them.
   - Replace the simulator's channel model with measured numbers. Cost: about $30–100.
5. **Flight.** 3–5 small drones indoors with flow decks, flying the simplest task that passed steps 1–3.

Risks:

- **Sim-to-real.** Flow drift, plume physics and radio interference near motors.
- **Payload and flight time.** Small drones carry little; two ESP32s, UWB, a flow deck and 2–3 accessories need a mid-size frame. Expect 5–10 minutes of flight.
- **Safety and regulation** for indoor and outdoor flight.
- **ESP-Drone's maintenance status.**

## Step 1 result (G13, `docs/results/r14_swarm_talk.md`)

**Radio does not bind.** Silent 73.4%, 1 B at 30% loss 94.9%, 2 B 97.6%, 8 B 97.6% (10 × 300 episodes). Talking matters (+24 points, a third faster), but a single-source report fits in 2 bytes.

Implication: in this swarm, learning would have to earn its place through coordination that is hard to hand-design, not compression. Examples are several targets with role allocation, obstacles, and drone or radio failures. Bandwidth binds only for rich content such as maps, images or very large swarms.

## Several sources (probe, `docs/results/r14_multi_probe.md`)

With 3 sources, coordination becomes the bottleneck:

| System | All three found in 2 min |
|---|---|
| Tuned hand-designed gossip swarm | 61.9% |
| Sensing-limited oracle (sources become known exactly at the first sniff, cameras split greedily, gas drones move on) | 93.1% |
| Silent | 29.9% |

The radio still does not bind (2 B = 8 B). This is the first good sign for *learned coordination* in the swarm; whether a learned policy closes the 31-point gap is the next (CPU-heavy) test.

## Step 2 result: learned per-role brains (`docs/results/r14_brains.md`)

**Good sign.** One small network per role (gas, camera), trained with evolution strategies on 3 leaks, decides where each drone heads; the radio and memory stay the hand design's.

| Leaks | Hand design | Learned brains (16 runs) | Sensed oracle |
|---|---|---|---|
| 1 | 96.6% | 97.0% | 98.6% |
| 2 | 85.4% | 91.3% | 96.9% |
| 3 (trained) | 62.5% | 78.3% (+15.9) | 93.2% |
| 4 | 33.6% | 58.0% (+24.4) | 87.9% |

The brains close half the coordination gap at 3 leaks and gain most on 4 leaks, which they never saw.

## Step 3 result (a): the brains survive worse conditions (`docs/results/r14_robust_step2.md`)

**Robust.**
- **Setup:** the step-2 brains against the hand design, retuning neither side, 3 leaks and 3,000 episodes per condition.
- **Bar (fixed in advance):** a gain with the CI above 0 in every harder condition.

| Condition | Hand design | Step-2 brains (gain) | Sensed oracle |
|---|---|---|---|
| nominal (trained) | 62.5% | 78.3% (+15.9 ± 2.0) | 93.2% |
| radio loss 50% | 63.4% | 78.0% (+14.6 ± 1.7) | 93.2% |
| radio loss 70% | 60.4% | 77.3% (+17.0 ± 2.4) | 93.2% |
| drift ×3 | 51.4% | 73.0% (+21.6 ± 2.5) | 91.8% |
| weak gas sensor | 50.9% | 75.1% (+24.2 ± 1.8) | 92.0% |
| narrow plume | 66.2% | 71.6% (+5.4 ± 1.7) | 82.0% |
| wide plume | 10.4% | 67.5% (+57.1 ± 1.5) | 46.5% |
| 1 drone down mid-mission | 54.0% | 71.4% (+17.4 ± 2.4) | 90.5% |
| 2 drones down mid-mission | 40.7% | 61.6% (+21.0 ± 2.5) | 86.0% |
| 4 drones (2 gas + 2 camera) | 36.9% | 56.5% (+19.6 ± 1.8) | 82.6% |
| 40 m arena | 30.3% | 45.6% (+15.3 ± 1.3) | 64.2% |

- **Where the gain grows:** when sensing or positions get worse (drift, weak sensor, lost drones).
- **Where it is smallest:** a narrow plume. There, finding the smell at all limits everyone, and the oracle drops too.
- **Wide plume:** the hand design's fixed smell threshold fails. The oracle's fixed threshold fails as well, so it is a bound only near its design point. The brains learned to cope.
- **Caveat:** the oracle is not told which drones went down.

## Step 3 result (b): randomised training adds little (`docs/results/r14_robust.md`)

**Setup:** 16 brains trained on a random condition every generation, with the same budget as step 2.
**Bar:** beat the step-2 brains on average over the harder conditions, CI above 0, and lose at most 3 points at nominal.
**Result:** passes, but the gain is tiny.

| | Gain over the step-2 brains |
|---|---|
| average over the harder conditions | +0.6 ± 0.2 points |
| nominal | 0.0 |
| wide plume (most of the gain) | +4.6 (72.1% vs 67.5%) |
| 4 drones (outside the randomised range) | −0.6 |
| 40 m arena (outside the randomised range) | −0.9 |

**What it means:**
- The step-2 brains were already robust.
- Randomisation costs nothing and helps where the plume is unusual, so flight brains can train with it.
- The bar should have set a minimum effect size; with no minimum, a 0.6-point gain counted as a pass.

## Bench preparation: the brains in int8 (`docs/results/r14_int8.md`)

**Pass.** The trained swarm brains lose nothing in int8. At nominal the int8 brains score −0.1 ± 0.6 points against float (step 2) and +0.3 ± 0.6 (randomised). At 70% radio loss and with 2 drones down they stay within ±0.3.

`hardware/esp32/swarm_brains_int8.h` holds the best-validation randomised brain (run 108) as int8 C arrays, 4 KB:
- the feature list and how to quantise the inputs;
- the per-layer integer recipe.

The bench step needs that header, an ESP32-S3, and a loop that builds the 15 features per candidate.

## R17: a leader that plans the person search (ladder step 3)

OpenSwarm's leader must turn "search the north field" into a plan:
- how high the K210 scouts sweep;
- how far apart their lanes are;
- the detection threshold;
- how many drones stay back to confirm detections, at what height and for how many looks.

The best plan depends on the mission. Higher sweeps cover more ground on one ~5 min battery, but people shrink in pixels, so recall falls. A low threshold finds more people but also more false alarms. Confirming costs drones.

**Setup.** `envs/swarm/mission.py` simulates one sortie (lanes, battery, a detection curve with false alarms, confirmers flying to candidates). Score = found − 2 × wrong find.

**Systems compared on 40 held-out missions:**
- the OpenSwarm default (SEARCH_GRID at 5 m, the first detection ≥ 0.7 is the find);
- a hand two-tier plan;
- one ES-tuned plan for all missions (blind);
- an ES-trained planner that reads the mission (conditioned);
- per-mission tuning, which shows the room.

**Bars, fixed in advance:**
- room: per-mission tuning beats the blind plan by ≥ 5 points;
- good sign: the conditioned planner beats the blind plan by ≥ 3 points, CI above 0.

**Caveat:** the detector's curve and false-alarm rate are ranges, not yet measurements. Once R15b reports, its measured curve becomes a held-out detector. The default plan's wrong finds come straight from the assumed false-alarm rates.

## Step 1 pilot (before pre-registration; 400 episodes per condition, seeds outside the experiment)

The pilot set the design and already points to an answer:

| Pilot result | Value |
|---|---|
| Silent swarm, 20 m arena | 95% success: talking cannot help |
| Silent swarm, 30 m arena (chosen) | 69% success |
| 8 B lossless, search radius tuned | 97.5% |
| 1 B at 30% loss, radius tuned | ~93%, about 84% of the talk gain |

- A camera's search radius matters more than position precision. A coarse code must widen the search to its cell, or it is worse than silence (55% before that fix).
- Expected verdict: talk matters, but on a single-source search **the radio does not bind**. If so, learned compression needs a richer task (step 2's candidates). The centre, translator and monitor pieces (steps 3–4) do not depend on the radio binding.

### R17 result: good sign (`docs/results/r17_mission.md`)

Results on 40 held-out missions × 2,000 episodes. Score = found − 2 × wrong find, in points.

| System | Score | Found | Wrong find |
|---|---|---|---|
| OpenSwarm default | −69.8 | 33.9% | 51.9% |
| hand two-tier | 38.2 | 46.6% | 4.2% |
| one tuned plan (16 runs) | 46.6 | 53.7% | 3.6% |
| planner that reads the mission (16 runs) | 64.6 | 69.0% | 2.2% |
| per-mission tuning | 69.1 | 74.0% | 2.4% |

- **Planner vs one plan:** +18.0 ± 6.1 points, capturing about 80% of the room (+22.6).
- **Consistency:** every planner run beats every one-plan run.
- **Plans chosen:** the planner sweeps between 3 and 17 m (averaged over runs); the one tuned plan always sweeps at 4.6 m.
  - It goes higher over larger areas: correlation +0.62 with area.
  - It goes higher with better detectors: correlation −0.5 with px50, where lower px50 is better.
  - With many false alarms it confirms with more looks at a high threshold.
- **Why the camera matters:** the detector sets the best altitude together with the area, so the camera is worth choosing on purpose (the hardware is ours to change).
- **Next:** R15b's measured curve as a held-out detector, and candidate modules (K210 at 224 px; Grove Vision AI V2; a MaixCAM-class Linux camera with an NPU) compared by search score after a weight and power cost.

### R17 step 1b: not robust to a walking person (`docs/results/r17_robust.md`)

**Setup:** the trained planners, not retrained, in 7 worse conditions.

**Where the planner keeps a clear lead over the one plan:**
- headwind: +17.4
- cold battery: +17.0
- areas ×1.6: +14.0
- 10 drones: +13.6

**Where it falls short of the bar:**
- a person walking 1.2 m/s: +3.1 ± 4.2
- 3 m GPS error: +2.2 ± 3.9

**Why every confirming plan collapses for a walking person:**
- At 0.5 m/s, finds drop from 69% to 6%.
- Confirmers hover over where the person was 30–60 s earlier.
- OpenSwarm's default, which never confirms, still finds 29%, but at 52% wrong finds.

**Next (R17 step 2):**
- **Hand-off:** the scout that detects a person keeps it in view, as person-follow does, and the confirmer comes to the latest sighting.
- **Training:** on missions that include walking people and worse GPS.
- **Scoring:** a find radius that grows with the GPS error.

### R17 step 2: hand-off, not yet (`docs/results/r17b_handoff.md`)

**Setup:** version 2 missions, half with a walking person, plus worse GPS, wind and battery variation; 16 + 16 planners.

**Scores on 40 held-out missions:**

| System | All | Walking | Still |
|---|---|---|---|
| one plan, v2 | 17.9 | 18.3 | 17.3 |
| planner that reads the mission, v2 | 29.3 | 29.3 | 29.3 |
| step-1 planner (no hand-off) | 18.4 | 2.7 | 39.6 |
| per-mission tuning | 38.7 | 34.6 | 44.1 |

- **Met:** the planner beats the one plan (+11.4) and fixes walking people (+26.6 over step 1).
- **Missed:**
  - a fixed 60 s hand-off adds +9.2 against a +10 bar;
  - on still-person missions the new planner is 15 points below step 1. It hedges because it cannot tell whether the person moves.
- **Next:** "may be moving" as a mission input from the operator (the ground station can ask), and a leader that switches to the hand-off when re-sightings move.

### R17 step 3: the moving hint does not help (`docs/results/r17c_hint.md`)

**Not yet.**
- **Does the planner use the hint?** Yes: it holds a sighting 26 s when told "still" and 95 s when told "may be moving".
- **Does the hint help?** Barely: 29.4 with a perfect hint against step 2's 29.3. On still-person missions it stays 15.6 points below the step-1 planner.
- **So step 2's "hedging" explanation was wrong.** The broad-trained planner sweeps higher (10.6 vs 6.8 m) and keeps more confirmers (39% vs 24%) even when told the person is still. Per-mission tuning finds better plans.
- **Next:** a larger planner (32 hidden units, 6,000 generations), judged against the same bars. If that fails, keep separate planners and let the hint pick.

## R15b result: the scout brain from a drone's view (`docs/results/r15b_aerial_scout.md`)

**Both bars pass.**
- Fine-tuning on VisDrone beats the COCO brain by +11.7 AP50 points (width 0.5).
- Starting from COCO beats training from scratch by +2.1.

Per-frame recall of the width-0.5 brain, by person size in the 224-px tile. Altitudes are rough, for the OV2640 at ~66/h px per person.

| Threshold | < 8 px (above ~8 m) | 8–16 px (~4–8 m) | 16–32 px (~2–4 m) | 32+ px | False detections per tile |
|---|---|---|---|---|---|
| 0.3 | 22% | 47% | 67% | 81% | 0.90 |
| 0.5 | 11% | 29% | 45% | 61% | 0.10 |
| 0.7 (OpenSwarm) | 2% | 5% | 7% | 24% | 0.002 |

**Consequences:**
- OpenSwarm's 0.7 threshold leaves the scout nearly blind.
- Search needs a lower threshold plus confirmation, which is R17's two-tier tactic.
- The measured curve is flatter than R17 assumed (half recall near 27 px), so R17's absolute numbers are optimistic.
- More pixels per person is the lever, through a higher-resolution camera module or lower sweeps.

### R17 step 3b: a bigger planner does not help (`docs/results/r17d_hint.md`)

**Setup:** twice the hidden units and twice the generations.
**Result:** no better. It stays 18 points below the step-1 planner on still people, and validation peaks by generation 600.
**What it means:** one planner trained by evolution over broad missions stalls. Next is planning at mission time: the ground station tunes the plan in the simulator for the actual mission just before launch.

### R17 step 5: plan at mission time (`docs/results/r17f_mission_time.md`)

**Not yet, by 0.2 points**, but the best design so far.

| Planner | 85% walking hint | Still person, hint "still" |
|---|---|---|
| mission-time tuning (30 s on one core) | **34.1** | **63.1** |
| step-2 planner | 29.3 | 49.2 |
| step-1 planner | 18.4 | 64.6 |
| tuning knowing the person (reference) | 38.7 | — |

**Recommendation for OpenSwarm's leader:** tune the plan on the ground station just before launch, warm-started from a learned planner. Use the learned planner alone only when there is no time to tune.

### R17 step 4: the scout brain's specification (`docs/results/r17e_requirements.md`)

**Setup:** R15b's measured brain in the search simulator, with the operator confirming candidates. Results are found / wrong finds.

| False detections | Tuned, 1 frame per look | Tuned, 3 frames per look |
|---|---|---|
| as measured | 25% / 6.4% | 41% / 9.9% |
| 10× fewer | 49% / 4.4% | **72% / 4.5%** |
| 30× fewer | **62% / 4.6%** | 81% / 3.2% |

**Specification:** cut false detections 10–30×, the higher figure if consecutive frames carry little new information.

**Cheapest route:** require a detection to persist over consecutive frames at the same spot before reporting it. Measuring how often false detections persist needs drone video (VisDrone video set).

## R15c probe: a 448-px input (`docs/results/r15c_aerial448.md`)

**Pixels on the person hold:** recall per size bin at 0.5 is within 4 points of the 224 brain's.

| Brain | Recall 0–8 / 8–16 / 16–32 / 32+ px | False detections per tile |
|---|---|---|
| 448 | 12 / 29 / 47 / 66% | 0.26 per 448 tile |
| 224 (R15b) | 11 / 29 / 45 / 61% | 0.10 per 224 tile |

**What it means for a camera:** a 448 camera with the same lens sees a person at altitude h as the K210 sees it at h/2. But each look raises about 2.7× the false detections. R17 step 4 with this brain weighs the two.

### Camera choice (R17 step 4 with the R15c brain)

Tuned plans, operator confirms; found / wrong finds.

| False detections | K210 224, 1 / 3 frames per look | 448-px camera, 1 / 3 frames per look |
|---|---|---|
| as measured | 25% / 41% found | 41% / 54% found |
| 10× fewer | 49% / 72% found | 71% / 85% found |

**Recommendation:** a camera module whose NPU runs the scout brain at 448 px (MaixCAM-class, about 1 TOPS), plus a persistence filter that cuts false detections about 10×. Bench check: frame rate at 448 on the chosen module, and whether a K210 can keep up.

## R17 step 6: the recommended search system (`docs/results/r17g_system.md`)

**Pass, narrowly.** Realistic missions: half with a walking person; GPS error, wind and battery vary.

| Setup (mission-time tuning, operator confirms) | Found | Wrong finds |
|---|---|---|
| K210, brain as measured | 10% | 6.6% |
| K210, false alarms cut 10× | 36% | 4.2% |
| 448 camera, as measured | 16% | 13.4% |
| **448 camera, false alarms cut 10×** | **52%** | **4.8%** |
| same, 1 frame per look | 35% | 4.4% |
| OpenSwarm default (448, cut 10×) | 19% | 72.9% |

**What a working person search needs (all of it):**
- a 448-px camera module;
- an on-drone filter cutting false alarms about 10×;
- operator confirmation from the drone's view;
- a plan tuned per mission against the measured detector.

**What to measure first:**
- frames per second that carry new information (drone video);
- how far a persistence filter cuts false alarms;
- operator accuracy;
- 448-px frame rate on the module.

### Which assumption matters most (R17 step 6b, `docs/results/r17h_sensitivity.md`)

Change in the share found from the recommended system's 52.3%, moving one assumption at a time:

| Variant | Found | Wrong finds |
|---|---|---|
| operator accepts 10% of empty views | −20.0 | 11.1% |
| 1 frame per second of video | −17.0 | 4.4% |
| filter keeps 60% of detections | −13.5 | 5.2% |
| false alarms cut 5× | −12.6 | 5.4% |
| operator accepts 5% of empty views | −12.2 | 7.1% |
| person 0.28 m | −8.1 | 5.1% |
| operator sees a person in 50% of looks | −5.0 | 4.4% |
| false alarms cut 20× | +11.2 | 4.1% |

**Measure first:**
- the operator's false-accept rate on drone images (a labelling session);
- independent frames per second (video);
- a persistence filter's cut and its loss of detections (video).

## R15d: more empty tiles in training (`docs/results/r15d_negatives.md`)

**Fail.** Training on every empty tile cuts false detections at matched recall to 0.72× (bar 0.5×; the search needs about 0.1×). Keep the recipe as a free improvement, but the big cut has to come from evidence across frames: persistence over consecutive frames, or a second, closer look.

## R15e: the operator, measured (`docs/results/r15e_operator.md`, `docs/results/r17h_operator.md`)

One operator labelled 200 flagged spots in a median of 0.75 s each. On the clear cases they said "person" for:
- 92% of people;
- 23% of non-people (bins, bollards, poles, roof tiles).

With those rates, the recommended search system finds 25% of people (was 52%), and 20% of its finds are wrong (was 4.8%).

**The quick single look is the weakest link.** Confirmation must change:
- a closer look from the confirming drone;
- several frames or a clip;
- a few seconds to decide;
- "can't tell" triggers another look.

**Next measurement:** a careful-mode labelling session.

## R15f: a careful look by the operator (`docs/results/r15f_careful.md`, `docs/results/r17h_careful.md`)

**Not enough.** 200 new spots, each with a close-up and a wider view, answer buttons locked for 2 s (median answer 2.3 s). On the clear cases the operator said "person" for:
- 90% of people;
- 8% of non-people [3, 21], down from 23%.

With those rates the recommended search system finds 36% of people (quick look 25%, assumed operator 52%), and 10% of its finds are wrong (quick look 20%, the limit 5%).

**Slowing down helps but cannot close the gap alone.** Fewer false alarms must reach the operator:
- persistence over several video frames (needs VisDrone-VID);
- a closer look from the confirming drone.


## R19: an open-weight confirmer, as is and fine-tuned (pre-registered 2026-11-03, goal G15)

**Question.** The scout brain is our own sub-megabyte network, and the ground station's LLM (qwen2.5-coder:14b) is used as is. Can an open-weight vision model, with its weights edited for drone footage, look at each detection before the operator and remove the false alarms R15e/R15f showed the operator cannot? And does editing the weights matter against using the model as is?

**Where it would run.** A 2B vision-language model needs the ground station's GPU (the scout sends the 144-px view of each detection, about 5 KB as JPEG). DINOv2-small (22M parameters) also fits a companion computer on the confirming drone (Jetson or Raspberry Pi class).

**Set-up.**
- Candidates: R15b's width-0.5 brain on VisDrone-DET frames at native scale (merged tiles), the 144-px view around each detection, labelled by R15b's centre rule. Train frames at threshold 0.4, at most 12 per frame; val and test-dev at 0.5.
- Fit on train, select on val (spots within 24 px of an R15e/R15f operator spot are left out), then score test-dev and the 400 operator spots once.
- The operator spots carry R15b's label, Claude's visual check of the false alarms (81 clear non-person, 53 person, 66 ambiguous) and the operator's answer, so the model is compared with the operator on the same spots.
- Models: DINOv2-small with a linear head on frozen features (weights kept) or fully fine-tuned (3 seeds); Qwen3-VL-2B-Instruct asked "is there a person at the exact centre?" as is, or with LoRA adapters trained on the same question (1 seed). Both are Apache-2.0.

**Read-out, fixed before any test score.** Each model's threshold keeps 95% of the val people. The chosen model is the edited one with the lowest val false-alarm pass.
- PASS if it passes ≤ 10% of the 81 clear non-person false alarms (R17 step 4's 10× cut) and keeps ≥ 90% of the 200 matched people among the operator spots.
- Editing matters if, on test-dev, the edited model passes at most half the false alarms of the same model unedited, with the frame-bootstrap upper bound of the ratio below 1.
- KILL if the chosen model passes ≥ 30% of the clear non-person false alarms.
- Measured with no bar: the model chained in front of the operator, and those rates in R17 step 6's search.

### R19 result: pass (`docs/results/r19_confirmer.md`, `docs/results/r17h_confirmer.md`)

**A fine-tuned open-weight model clears the 10× bar on clearly empty spots; the same model used as is does not.** On the 81 clear non-person false alarms among the operator spots, the share each one lets through (each keeps 93–96% of people):

| Who | Clear non-persons passed |
|---|---|
| Operator alone | 16.0% |
| DINOv2 frozen | 27.2% |
| DINOv2 fine-tuned (chosen) | 6.2% [3, 14] |
| Qwen3-VL as is | 40.7% |
| Qwen3-VL with LoRA | 2.5% |

On test-dev, editing cuts the false alarms passed to 0.56× (DINOv2) and 0.44× (Qwen) of the unedited model's.

**The errors overlap with the operator's.** The operator accepted 4 of the 5 clear non-persons the model passes. Chained, the two let through 4.9%. The search finds 40% with 7% wrong finds (careful operator alone 36% / 10%), still over the 5% limit. The next cut must come from evidence whose errors differ: persistence over video frames (R15g) or a closer look from the confirming drone.

**Follow-ups:**
- Distil the fine-tuned confirmer into the K210 scout brain, as a teacher on unlabelled frames.
- Put the Qwen LoRA and DINOv2 together.
- Fine-tune the ground station's LLM on the R16 dev data (G14b missed narrowly). G3b's lesson applies there: train on real phrasing, not templates.

## R20: a closer look from the confirming drone (pre-registered 2026-11-04, goal G16)

**Question.** R19's confirmer and the operator fail on the same false alarms, and two models on the same view fail together (DINOv2 and Qwen correlate 0.85; together they barely beat one). In the mission the confirming drone hovers far lower than the scout sweeps. Does that closer look fail on *different* false alarms, so that a second check is worth something?

**Set-up.**
- VisDrone2019-DET stands in for both views. The scout's view is each frame reduced 2× (the scout twice as high); the closer look is the native frame (the confirmer at half the height: the same 144-px window, half the ground, twice the detail).
- R15b's brain runs on the reduced frames. Each candidate gets the 144-px view around it in both resolutions and R15b's label. Both views come from one photo, so pose, light and blur are shared: this measures what detail alone adds. A real second look (another angle, another moment) can only be less correlated.
- DINOv2-small fine-tuned per view (R19's recipe: 10 epochs, 336 px), two seeds each.
- Control chain: the scout's view checked by two models (seeds 0 and 1). Closer-look chain: the scout's view (seed 0), then the native view (seed 0). Replicate: seed 1 of each. Thresholds on val: each model keeps the same share of val people, and the pair keeps 95%.

**Read-out, fixed before any test score.** On test-dev false alarms (VisDrone labels):
- PASS if the closer-look chain passes ≤ 0.5× what the control passes, with the frame-bootstrap upper bound below 1, the replicate also ≤ 0.5×, and the closer-look chain keeps ≥ 90% of test-dev people.
- KILL if it passes ≥ 0.8× of the control's.
- Measured with no bar: each model alone, how often the second model passes what the first passed, score rank correlations, and Claude's blind visual check of up to 150 test-dev false alarms passed by either chain (random order, chain hidden). Unlabelled people count as false alarms for both chains, which pulls the ratio toward 1 (against a pass).

**Result (2026-11-04): KILL.** The closer-look chain passes 0.86× [0.82, 0.90] of the control's test-dev false alarms (18.3% against 21.2%; replicate 0.84×), with 93.6% of people kept. On its own the native view is the better model (val AUC 0.94 against 0.92), but its scores on false alarms correlate at 0.85 with the scout-view model's, and it passes two thirds of what that model passed. Claude's blind check of 150 of the passed false alarms found 111 people (57 inside a labelled box, counted twice by the matching rule; 54 not boxed), 27 unclear and 12 clearly not a person. The control passed all 12 of those and the closer look 3: a hint on too few spots. Dropping false alarms near labelled people leaves the ratio at 0.72–0.85×. So detail alone does not decorrelate the errors, and once a confirmer is in the chain, VisDrone's labels mostly measure label noise. Next: a blind check of all 744 passed false alarms (scored on the 594 unseen ones) settles the clear-case ratio; video persistence (R15g) is the other candidate.

## R20b: the closer look on clear non-persons (pre-registered 2026-11-04, goal G16b)

**Question.** R20's labelled false alarms are mostly people (111 of 150 checked), so its KILL mostly measured label noise. On the 12 clear non-persons in that check, the control chain passed all 12 and the closer look 3. Is that real: does the closer look cut the clearly empty spots the chain lets through at least in half?

**Set-up.** R20's scores and val thresholds, nothing refit or re-scored. Every test-dev false alarm (VisDrone labels) that any of the three chains passes: 744, of which 594 R20 did not check. Those 594 plus 36 R20-checked spots (for agreement, not scored) go on sheets in one random order: the closer look's centre at 4x and the whole closer look with the centre marked; the chain is never shown. Claude subagents label each spot: is a person (standing, walking, sitting or riding) at the marked centre? person / not_person / ambiguous.

**Read-out, fixed before any label.** Among the 594 new spots labelled not_person:
- PASS if the closer-look chain passes <= 0.5x what the control passes, frame-bootstrap upper bound < 1, and the replicate also <= 0.5x.
- KILL if it passes >= 0.8x.
- Measured with no bar: per chain, the share of its passed false alarms that are people; agreement with R20's labels on the 36 calibration spots.

**Result (2026-11-04): PASS.** The 594 new spots: 457 people, 96 unclear, 41 clearly not a person. Of the clear non-persons, the control passes 35, the closer look 15 (0.43x [0.24, 0.66]) and the replicate 9 (0.26x [0.10, 0.43]); with R20's 150 spots added, 0.38x [0.24, 0.56]. The rejected spots are rigid things that look like a person from twice as high: lamp posts, cones, bins, traffic lights, tables with chairs. On unclear spots the cut is 0.68x. The subagents agreed with R20's labels on 81% of the 36 mixed-in spots, and Claude's own blind re-check of the 41 not-a-person labels agreed on 37-39. So the closer look is worth flying: it removes the truly empty spots a second model on the scout's view lets through, which the labelled rate (0.86x) hides under unlabelled people. In R17h's mission simulation (`docs/results/r17h_closer_look.md`), R19's confirmer plus the closer look gives 51.9% found with 4.7% wrong finds (R19 alone: 40.4% / 7.0%); at the cut's upper bound 47.6% / 5.6%, at the replicate's 56.2% / 3.7%. The search meets its limits at the measured cut, with no slack; video persistence (R15g) is the next cut.

## R21: the ground station's language model with its weights edited (pre-registered 2026-11-04, goal G17)

**Question.** The ground station uses qwen2.5-coder:14b as is and missed G14b's exact-match bar narrowly (88.9%, bar 90%). R19 showed that editing open weights is what made the vision confirmer work. Does the same hold for the language side: can a smaller open-weight model, fine-tuned on how operators really talk, reach the bars the unedited 14B missed?

**Why this can differ from G3b.** R7a's distilled students failed on hand-written requests because their training text was template-like. Here five writers with different styles (radio calls, chatty hobbyists, voice transcripts, professionals, chat-app beginners) write the training requests from the label guide alone.

**Set-up.**
- Qwen2.5-Coder-7B-Instruct in 4-bit with QLoRA, trained only on the reply, with the exact prompt the 14B gets, inside the unchanged v5 ground-station pipeline (literal check, placement check, retries).
- Training data checks: each label must pass the spec and the literal check, no text may be close to a dev or sealed text (4-gram containment < 0.5), and the deployed 14B reads every training request; its disagreements are reviewed by hand and labels that break the guide are dropped.
- Epoch (of 3) picked on R16 dev. The same 7B unedited, with the same prompt and decoding, is the "as is" system.

**Read-out, fixed before any sealed run.** G14b's sealed set, once per new system:
- PASS if the tuned 7B reaches exact ≥ 90% and wrong commands ≤ 3% with the Wilson upper bound ≤ 6%.
- Editing matters if its exact match is ≥ 10 points above the 7B as is, with McNemar p < 0.05.
- KILL if exact < 80% or wrong commands > 10%.
- Measured with no bar: the 14B's G14b result on the same items, asks and refusals, latency.

**Result (2026-11-04): PASS.** On G14b's 150 sealed requests the 7B + QLoRA gets 91.1% [83, 95] exact with 2.0% [1, 6] wrong commands; the same 7B as is gets 81.1% and 3.3%, so editing adds 10.0 points (McNemar p = 0.035, exactly at the bar). The deployed 14B as is got 88.9% / 2.7%. Paired with the 14B, exact match is level (7 items against 5), but the tuned 7B gets the right outcome on 19 items where the 14B fails, against 8 (p = 0.052); 11 of them are unclear requests the 14B refused or ran, where the 7B asks (it asks on all 30). Its 12 misses are 7 questions about place names not tied to the site map (schoolyard, parking lot, motorway, forest), 3 compound requests that should be refused but became their first half ("then return home", "but don't move"), a wrong refusal and a failed reply. Training needed one fix: peft's 4-bit prep upcast the embedding and output layers to fp32 and ran the card out of memory; they now stay in bf16 and only the reply's logits are computed.

## R22: the scout brain trained on confirmer-cleaned labels (pre-registered 2026-11-04, goal G18)

**Question.** R20b found that about three in four of VisDrone's "false alarms" that get past the confirmers are people the labels miss. The scout brain (R15b) learned from those labels, so part of its training said people are background. If an out-of-fold confirmer puts the missing people back, does the scout raise fewer *clearly empty* false alarms at the same share of labelled people found?

**Set-up.**
- R19's 56,538 train candidates are split by sequence into two folds. DINOv2-small (R19's recipe) is trained on one fold and scores the other, so every candidate is scored by a model that never saw its sequence.
- A candidate VisDrone calls a false alarm, outside every person box, becomes a person box when its out-of-fold score reaches the threshold that passes 80% of that fold's labelled people.
- The scout is retrained with R15b's recipe on R15b's own training tiles with those boxes added (seeds 0 and 1). The control is R15b's fine-tuned seeds 0 and 1.
- All four scouts run over val and test-dev frames at native scale. Thresholds are set on val so each finds the share of labelled people that control seed 0 finds at 0.5.
- 150 random test-dev false alarms per scout go on blind sheets (scout hidden), labelled by Claude subagents.

**Read-out, fixed before any pseudo-label exists.**
- PASS if clear non-person false alarms per frame, cleaned over control (seeds pooled), is <= 0.7x with the frame-bootstrap upper bound < 1, each seed pair is <= 0.8x, and test-dev labelled people found stay within 2 points of control.
- KILL at >= 0.9x.
- Measured: blind check of 100 pseudo-positives and 100 rejected candidates (label precision), labelled false alarms per frame, and people among the false alarms.
- Control scouts were trained on 2026-10-29 with the same recipe; the cleaned ones reuse the same tiles and images, so only the boxes differ.

