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

## R17: a leader that plans the person search (ladder step 3, running)

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
