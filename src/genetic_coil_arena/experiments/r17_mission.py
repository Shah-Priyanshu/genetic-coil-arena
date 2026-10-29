"""R17 (ladder step 3): a leader that plans the person search from the mission.

OpenSwarm's leader turns a search request into a plan for the swarm: how high the
scouts sweep, how far apart their lanes are, the detection threshold, how many drones
stay back to confirm and how they confirm (``envs/swarm/mission.py``). The best plan
depends on the mission: the area, the number of drones, and how good the scout brain
is (its detection curve and false-alarm rate). This step asks whether a learned planner
that reads the mission beats one tuned plan for every mission.

Systems, all scored on the same people (seeds):

- ``OpenSwarm default``: SEARCH_GRID at 5 m, the first detection >= 0.7 is the find;
- ``hand two-tier``: sweep at 8 m at 0.5, one or two confirmers at 4 m (``hand_two_tier``);
- ``blind``: one plan for all missions, tuned with ES on random training missions;
- ``conditioned``: an MLP from the mission's features to the plan, trained the same way
  (same start, budget and missions; ``PlannerSettings``);
- ``per-mission tuned``: ES on each test mission alone, on its own tuning seeds (the
  room a planner could win; not a strict bound).

Missions (``sample_mission``): a 60-240 m by 60-240 m area, 3-8 drones, one 300 s
battery, and a detector with ``px50`` 6-20 px, slope 1.5-3.5 and 0.0005-0.03 false
detections per look at 0.5 (log-uniform). R15b's measured curve is a later held-out
detector. Score per episode: found - 2 x wrong find; reported in points (x 100).

Bars (fixed before any training run), on the 40 test missions with 2,000 episodes each,
paired by mission (t CI over missions; learned systems averaged over their runs first):

- **room:** per-mission tuned beats blind by >= 5 points, CI above 0;
- **good sign:** conditioned beats blind by >= 3 points, CI above 0.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..envs.swarm.mission import (
    FOUND_M,
    LOC0,
    OPENSWARM_DEFAULT,
    Detector,
    Mission,
    Plan,
    hand_two_tier,
    run_missions,
)
from ..evaluation import mean_ci
from ..learning import MLP, EvolutionStrategy

PLAN_RANGES = (
    ("sweep_alt", 3.0, 25.0),
    ("overlap", 0.0, 0.6),
    ("sweep_thr", 0.3, 0.95),
    ("confirm_frac", 0.0, 0.67),
    ("confirm_alt", 2.0, 12.0),
    ("confirm_thr", 0.3, 0.95),
    ("confirm_looks", 1.0, 10.0),
)
N_FEATURES, N_PLAN = 6, len(PLAN_RANGES)
# R17 step 2 (version 2): the plan adds the hand-off; missions add a walking person,
# GPS error, wind and battery, and the leader reads the last three
PLAN_RANGES_V2 = PLAN_RANGES + (("track_s", 0.0, 120.0),)
N_FEATURES_V2, N_PLAN_V2 = 9, len(PLAN_RANGES_V2)
TRAIN_BASE, VAL_BASE, TEST_BASE = 10_000_000, 1_500_000_000, 2_000_000_000
ORACLE_BASE, ORACLE_VAL_BASE = 500_000_000, 1_000_000_000
RUN_STRIDE = 20_000_000
N_TEST, N_VAL, TEST_EPISODES = 40, 24, 2000
ROOM_BAR, GAIN_BAR = 0.05, 0.03


def find_radius(loc_error: float) -> float:
    """Version 2: the ground team searches a radius that grows with the GPS error."""
    return FOUND_M + 2.5 * (loc_error - LOC0)


def sample_mission(rng: np.random.Generator, version: int = 1) -> Mission:
    w, h = rng.uniform(60.0, 240.0, 2)
    det = Detector(
        px50=float(rng.uniform(6.0, 20.0)),
        slope=float(rng.uniform(1.5, 3.5)),
        fa50=float(10 ** rng.uniform(math.log10(0.0005), math.log10(0.03))),
    )
    m = Mission(float(w), float(h), int(rng.choice([3, 4, 5, 6, 8])), 300.0, det)
    if version == 1:
        return m
    walk = float(rng.uniform(0.2, 1.5)) if rng.random() < 0.5 else 0.0
    loc = float(rng.uniform(1.0, 4.0))
    return replace(m, battery_s=float(rng.uniform(220.0, 320.0)), person_speed=walk,
                   loc_error=loc, speed=float(rng.uniform(3.0, 5.0)),
                   found_m=find_radius(loc))  # fmt: skip


def held_out_missions(version: int = 1) -> list[Mission]:
    rng = np.random.default_rng(1718 if version == 1 else 1728)
    return [sample_mission(rng, version) for _ in range(N_TEST)]


def val_missions(version: int = 1) -> list[Mission]:
    rng = np.random.default_rng(1719 if version == 1 else 1729)
    return [sample_mission(rng, version) for _ in range(N_VAL)]


def features(m: Mission, version: int = 1) -> np.ndarray:
    """What the leader knows: the area, the drones and the detector; version 2 adds
    the GPS quality, the wind (cruise speed) and the battery. Never the person's speed.
    """
    d = m.detector
    f = [(m.width - 150) / 90, (m.height - 150) / 90, (m.drones - 5) / 2.5,
         (d.px50 - 13) / 7, d.slope - 2.5, (math.log10(d.fa50) + 2.4) / 0.9]  # fmt: skip
    if version == 2:
        f += [(m.loc_error - 2.5) / 1.5, m.speed - 4.0, (m.battery_s - 270) / 50]
    return np.array(f, dtype=np.float32)


def n_features(version: int) -> int:
    return N_FEATURES if version == 1 else N_FEATURES_V2


def n_plan(version: int) -> int:
    return N_PLAN if version == 1 else N_PLAN_V2


def decode_plan(u: np.ndarray, drones: int, version: int = 1) -> Plan:
    ranges = PLAN_RANGES if version == 1 else PLAN_RANGES_V2
    lo = np.array([r[1] for r in ranges])
    hi = np.array([r[2] for r in ranges])
    v = lo + (hi - lo) / (1.0 + np.exp(-np.asarray(u, dtype=np.float64)))
    return Plan(sweep_alt=float(v[0]), overlap=float(v[1]), sweep_thr=float(v[2]),
                confirmers=min(drones - 1, int(round(v[3] * drones))),
                confirm_alt=float(v[4]), confirm_thr=float(v[5]),
                confirm_looks=int(round(v[6])),
                track_s=float(v[7]) if version == 2 else 0.0)  # fmt: skip


def encode_plan(plan: Plan, drones: int, version: int = 1) -> np.ndarray:
    v = [plan.sweep_alt, plan.overlap, plan.sweep_thr, plan.confirmers / drones,
         plan.confirm_alt, plan.confirm_thr, plan.confirm_looks]  # fmt: skip
    ranges = PLAN_RANGES if version == 1 else PLAN_RANGES_V2
    if version == 2:
        v.append(plan.track_s)
    u = []
    for x, (_, lo, hi) in zip(v, ranges):
        f = min(max((x - lo) / (hi - lo), 0.02), 0.98)
        u.append(math.log(f / (1 - f)))
    return np.array(u)


def mission_seeds(base: int, index: int, n: int) -> list[int]:
    return [base + index * 10_000 + e for e in range(n)]


@dataclass(frozen=True)
class PlannerSettings:
    conditioned: bool = True
    hidden: int = 16
    generations: int = 3000
    population: int = 48
    missions: int = 16  # per generation, fresh each generation
    episodes: int = 48  # per mission
    sigma: float = 0.1
    lr: float = 0.03
    eval_every: int = 50
    val_episodes: int = 200
    progress_s: float = 120.0
    version: int = 1


class Planner:
    """The blind planner is one plan vector; the conditioned one an MLP from
    ``features`` to it."""

    def __init__(self, settings: PlannerSettings):
        self.settings, self.version = settings, settings.version
        nf, npl = n_features(self.version), n_plan(self.version)
        self.net = MLP((nf, settings.hidden, npl)) if settings.conditioned else None
        self.n_params = self.net.n_params if self.net else npl

    def init(self, rng: np.random.Generator) -> np.ndarray:
        start = hand_two_tier(6)
        if (
            self.version == 2
        ):  # version 2 starts from the hand plan with a 30 s hand-off
            start = replace(start, track_s=30.0)
        u0 = encode_plan(start, 6, self.version)
        if self.net is None:
            return u0
        npl = len(u0)
        theta = self.net.init(rng, scale=0.5)
        theta[-npl:] = u0  # the output bias: every mission starts at the hand plan
        theta[-npl - self.settings.hidden * npl : -npl] *= 0.1
        return theta

    def plans(self, theta: np.ndarray, missions: Sequence[Mission]) -> list[Plan]:
        v = self.version
        if self.net is None:
            return [decode_plan(theta, m.drones, v) for m in missions]
        u = self.net.forward(theta, np.stack([features(m, v) for m in missions]))
        return [decode_plan(u[i], m.drones, v) for i, m in enumerate(missions)]


def mean_score(plans: Sequence[Plan], missions: Sequence[Mission],
               seeds: Sequence[Sequence[int]]) -> float:  # fmt: skip
    return float(np.mean([run_missions(m, p, s)["score"].mean()
                          for m, p, s in zip(missions, plans, seeds)]))  # fmt: skip


def train_run(run: int, settings: PlannerSettings) -> dict[str, Any]:
    planner = Planner(settings)
    theta0 = planner.init(np.random.default_rng([run, 17]))
    es = EvolutionStrategy(theta0, population=settings.population, sigma=settings.sigma,
                           lr=settings.lr, weight_decay=0.0, seed=run)  # fmt: skip
    val = val_missions(settings.version)
    val_seeds = [
        mission_seeds(VAL_BASE, i, settings.val_episodes) for i in range(N_VAL)
    ]
    best = (-np.inf, es.theta.copy(), 0)
    curve: list[dict[str, float]] = []
    start = last = time.perf_counter()
    for g in range(settings.generations + 1):
        if g % settings.eval_every == 0 or g == settings.generations:
            v = mean_score(planner.plans(es.theta, val), val, val_seeds)
            curve.append({"generation": g, "val": v,
                          "wall_s": time.perf_counter() - start})  # fmt: skip
            if v > best[0]:
                best = (v, es.theta.copy(), g)
            now = time.perf_counter()
            if settings.progress_s and (now - last >= settings.progress_s or g == 0):
                last = now
                kind = "conditioned" if settings.conditioned else "blind"
                print(f"[progress] {kind} run {run}: gen {g}/{settings.generations}, "
                      f"val {v:.3f}, best {best[0]:.3f} at {best[2]}, "
                      f"{now - start:.0f} s", flush=True)  # fmt: skip
        if g == settings.generations:
            break
        rng = np.random.default_rng([run, g, 1717])
        missions = [sample_mission(rng, settings.version)
                    for _ in range(settings.missions)]  # fmt: skip
        base = TRAIN_BASE + run * RUN_STRIDE + g * settings.missions * settings.episodes
        seeds = [[base + i * settings.episodes + e for e in range(settings.episodes)]
                 for i in range(settings.missions)]  # fmt: skip
        cands = es.ask()
        es.tell(np.array([mean_score(planner.plans(c, missions), missions, seeds)
                          for c in cands]))  # fmt: skip
    return {"run": run, "theta": best[1], "best_val": float(best[0]),
            "best_generation": best[2], "curve": curve,
            "seconds": time.perf_counter() - start, "n_params": int(theta0.size),
            "settings": asdict(settings)}  # fmt: skip


def tune_mission(index: int, generations: int = 300, population: int = 32,
                 episodes: int = 128, val_episodes: int = 1000,
                 version: int = 1) -> dict[str, Any]:  # fmt: skip
    """ES on one test mission's plan alone, from the default and the hand plan, on its
    own tuning seeds; the start whose result scores best on separate seeds wins.
    Version 2 also starts from the hand plan with a 30 s hand-off."""
    m = held_out_missions(version)[index]
    val_seeds = mission_seeds(ORACLE_VAL_BASE, index, val_episodes)
    best: tuple[float, np.ndarray, str] = (-np.inf, np.zeros(n_plan(version)), "")
    starts = [("default", OPENSWARM_DEFAULT), ("hand", hand_two_tier(m.drones))]
    if version == 2:
        starts.append(("hand-off", replace(hand_two_tier(m.drones), track_s=30.0)))
    for k, (name, plan) in enumerate(starts):
        es = EvolutionStrategy(encode_plan(plan, m.drones, version), population=population,
                               sigma=0.15, lr=0.05, weight_decay=0.0,
                               seed=index * 10 + k)  # fmt: skip
        for g in range(generations):
            seeds = [ORACLE_BASE + index * 1_000_000 + (k * generations + g) * episodes + e
                     for e in range(episodes)]  # fmt: skip
            es.tell(np.array([run_missions(m, decode_plan(c, m.drones, version),
                                           seeds)["score"].mean()
                              for c in es.ask()]))  # fmt: skip
        plan = decode_plan(es.theta, m.drones, version)
        v = float(run_missions(m, plan, val_seeds)["score"].mean())
        if v > best[0]:
            best = (v, es.theta.copy(), name)
    return {"mission": index, "u": best[1].tolist(), "val": best[0], "start": best[2],
            "version": version,
            "plan": asdict(decode_plan(best[1], m.drones, version))}  # fmt: skip


# ------------------------------------------------------------------ save, load, evaluate
def save_run(out: dict[str, Any], folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    kind = "conditioned" if out["settings"]["conditioned"] else "blind"
    prefix = "r17_planner" if out["settings"].get("version", 1) == 1 else "r17b_planner"
    stem = folder / f"{prefix}_{kind}_run{out['run']}"
    np.savez(stem.with_suffix(".npz"), theta=out["theta"])
    meta = {k: v for k, v in out.items() if k != "theta"}
    stem.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return stem


def load_runs(folders: Sequence[Path], kind: str,
              version: int = 1) -> list[dict[str, Any]]:  # fmt: skip
    prefix = "r17_planner" if version == 1 else "r17b_planner"
    runs = []
    for folder in folders:
        for meta in sorted(Path(folder).rglob(f"{prefix}_{kind}_run*.json")):
            d = json.loads(meta.read_text(encoding="utf-8"))
            with np.load(meta.with_suffix(".npz")) as z:
                d["theta"] = z["theta"]
            runs.append(d)
    return sorted(runs, key=lambda r: r["run"])


def load_oracle(folders: Sequence[Path], version: int = 1) -> dict[int, dict[str, Any]]:
    prefix = "r17_oracle_m" if version == 1 else "r17b_oracle_m"
    out = {}
    for folder in folders:
        for f in sorted(Path(folder).rglob(f"{prefix}*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            out[d["mission"]] = d
    return out


def system_plans(blind: Sequence[dict], cond: Sequence[dict],
                 oracle: dict[int, dict]) -> dict[str, list[list[Plan]]]:  # fmt: skip
    """Per system, one list of test-mission plans per run (one run for fixed plans)."""
    ms = held_out_missions()
    out: dict[str, list[list[Plan]]] = {
        "OpenSwarm default": [[OPENSWARM_DEFAULT] * len(ms)],
        "hand two-tier": [[hand_two_tier(m.drones) for m in ms]],
    }
    for name, runs in (("blind", blind), ("conditioned", cond)):
        if runs:
            out[name] = [Planner(PlannerSettings(**r["settings"])).plans(r["theta"], ms)
                         for r in runs]  # fmt: skip
    if oracle and len(oracle) == len(ms):
        out["per-mission tuned"] = [[decode_plan(np.array(oracle[i]["u"]), m.drones)
                                     for i, m in enumerate(ms)]]  # fmt: skip
    return out


def eval_cell(task: tuple) -> tuple[tuple, dict[str, float]]:
    """One (system, run, mission) cell on the mission's test seeds."""
    key, mission_index, plan, episodes = task
    m = held_out_missions()[mission_index]
    r = run_missions(m, plan, mission_seeds(TEST_BASE, mission_index, episodes))
    return key, {k: float(r[k].mean()) for k in ("score", "found", "wrong", "swept")}


def eval_tasks(plans: dict[str, list[list[Plan]]], episodes: int = TEST_EPISODES):
    return [((name, r, i), i, plan, episodes)
            for name, runs in plans.items() for r, run_plans in enumerate(runs)
            for i, plan in enumerate(run_plans)]  # fmt: skip


def summarise(cells: dict[tuple, dict], plans: dict[str, list[list[Plan]]]) -> dict:
    n = N_TEST
    per: dict[str, dict[str, np.ndarray]] = {}
    for name, runs in plans.items():
        per[name] = {k: np.array([[cells[(name, r, i)][k] for i in range(n)]
                                  for r in range(len(runs))])  # fmt: skip
                     for k in ("score", "found", "wrong", "swept")}
    out: dict[str, Any] = {
        "systems": {},
        "missions": [asdict(m) for m in held_out_missions()],
    }
    for name, d in per.items():
        out["systems"][name] = {
            "runs": len(plans[name]),
            **{k: float(v.mean()) for k, v in d.items()},
            "per_mission_score": d["score"].mean(0).tolist(),
            "run_scores": d["score"].mean(1).tolist(),
        }
    if "blind" in per:
        blind = per["blind"]["score"].mean(0)
        for name in per:
            if name != "blind":
                out["systems"][name]["minus_blind"] = mean_ci(
                    (per[name]["score"].mean(0) - blind).tolist())  # fmt: skip
        if "per-mission tuned" in per:
            room = out["systems"]["per-mission tuned"]["minus_blind"]
            out["room"] = {"diff": room, "pass": bool(room[0] >= ROOM_BAR
                                                      and room[0] - room[1] > 0)}  # fmt: skip
        if "conditioned" in per:
            gain = out["systems"]["conditioned"]["minus_blind"]
            out["good_sign"] = {"diff": gain, "pass": bool(gain[0] >= GAIN_BAR
                                                           and gain[0] - gain[1] > 0)}  # fmt: skip
    return out


def example_plans(plans: dict[str, list[list[Plan]]]) -> list[dict[str, Any]]:
    """Run-0 plans of every system on six telling held-out missions."""
    ms = held_out_missions()
    area = [m.width * m.height for m in ms]
    picks = {
        "smallest area": int(np.argmin(area)), "largest area": int(np.argmax(area)),
        "best detector": int(np.argmin([m.detector.px50 for m in ms])),
        "worst detector": int(np.argmax([m.detector.px50 for m in ms])),
        "fewest false alarms": int(np.argmin([m.detector.fa50 for m in ms])),
        "most false alarms": int(np.argmax([m.detector.fa50 for m in ms])),
    }  # fmt: skip
    return [{"why": why, "mission": i, "spec": asdict(ms[i]),
             "plans": {name: asdict(runs[0][i]) for name, runs in plans.items()}}
            for why, i in picks.items()]  # fmt: skip


def plan_table(plans: dict[str, list[list[Plan]]]) -> list[dict[str, Any]]:
    """Per held-out mission: its spec and each system's plan, averaged over runs."""
    keys = ("sweep_alt", "sweep_thr", "confirmers", "confirm_alt")
    out = []
    for i, m in enumerate(held_out_missions()):
        row: dict[str, Any] = {"mission": i, "width": m.width, "height": m.height,
                               "drones": m.drones, "px50": m.detector.px50,
                               "fa50": m.detector.fa50, "systems": {}}  # fmt: skip
        for name, runs in plans.items():
            row["systems"][name] = {
                k: float(np.mean([getattr(r[i], k) for r in runs])) for k in keys
            }
        out.append(row)
    return out


def _pts(x: float) -> str:
    return f"{100 * x:+.1f}"


def _plan_text(p: dict[str, Any]) -> str:
    text = f"sweep {p['sweep_alt']:.0f} m at {p['sweep_thr']:.2f}"
    if p["confirmers"]:
        text += (f", {p['confirmers']} confirm at {p['confirm_alt']:.0f} m, "
                 f"{p['confirm_looks']} looks at {p['confirm_thr']:.2f}")  # fmt: skip
    return text


def render(rep: dict[str, Any]) -> str:
    lines = ["# R17: does a leader that reads the mission plan the search better?", ""]
    for key, label, bar, what in (
        ("room", "Room", ROOM_BAR, "Tuning the plan per mission beats the one blind plan"),
        ("good_sign", "Good sign", GAIN_BAR,
         "The conditioned planner beats the one blind plan"),
    ):  # fmt: skip
        if key in rep:
            d = rep[key]["diff"]
            lines += [f"**{label}: {'PASS' if rep[key]['pass'] else 'FAIL'}.** {what} by "
                      f"{_pts(d[0])} ± {100 * d[1]:.1f} points (bar ≥ +{100 * bar:.0f}, "
                      "CI above 0).", ""]  # fmt: skip
    lines += [f"{N_TEST} held-out missions × {TEST_EPISODES} episodes; score = found − 2 × "
              "wrong find, in points; differences are paired by mission (95% CI).", "",
              "| System | Runs | Score | Found | Wrong find | Lanes flown | vs blind |",
              "|---|---|---|---|---|---|---|"]  # fmt: skip
    for name, s in rep["systems"].items():
        vs = s.get("minus_blind")
        lines.append(
            f"| {name} | {s['runs']} | {100 * s['score']:.1f} | {100 * s['found']:.1f}% "
            f"| {100 * s['wrong']:.1f}% | {100 * s['swept']:.0f}% | "
            + (f"{_pts(vs[0])} ± {100 * vs[1]:.1f}" if vs else "—")
            + " |"
        )
    if rep.get("examples"):
        lines += ["", "Plans chosen (run 0) on telling missions:", ""]
        for ex in rep["examples"]:
            sp, d = ex["spec"], ex["spec"]["detector"]
            lines.append(f"- **{ex['why']}** ({sp['width']:.0f} × {sp['height']:.0f} m, "
                         f"{sp['drones']} drones, px50 {d['px50']:.0f}, "
                         f"{d['fa50']:.4f} false/look):")  # fmt: skip
            for name, p in ex["plans"].items():
                lines.append(f"  - {name}: {_plan_text(p)}")
    lines += [
        "",
        "Reproduce: `python scripts/r17_mission.py report --from runs:r17-*`.",
        "",
    ]
    return "\n".join(lines)
