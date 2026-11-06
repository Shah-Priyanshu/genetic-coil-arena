"""Project dashboard (Streamlit): the research story, live runner, goal ladder, charts, reports.

    powershell -File scripts/start_dashboard.ps1        # own window, http://localhost:8501
    D:/venvs/dash/Scripts/streamlit run scripts/dashboard.py

The first tab tells the story from ``docs/research_log.md``. For every test it shows the
question, why it was run, what was done, the result, the verdict, what it means and its
chart, with the full report one click away. Keep that file current with every result.

It only reads files: the results worktree (``runner/state.json``, ``runs/*``) and ``docs/``
of this checkout. It never imports the project package, never writes, and never touches
the GPU. Set ``GCA_RESULTS_DIR`` if the results worktree is not next to this checkout.
The helpers above ``main`` use the standard library only, so tests import them without
Streamlit.
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
RESULTS = Path(
    os.environ.get("GCA_RESULTS_DIR", REPO.parent / "genetic-coil-arena-gpu-results")
)
DOCS = REPO / "docs"

CATEGORIES = {
    "pass": "#2e7d32",
    "mixed": "#f9a825",
    "kill": "#c62828",
    "pending": "#1565c0",
    "probe": "#6a1b9a",
}
LOG_FIELDS = (
    "Date",
    "Question",
    "Why",
    "What we did",
    "Result",
    "Verdict",
    "What it means",
    "Report",
    "Chart",
)


# ------------------------------------------------------------------ data helpers
def parse_goals(text: str) -> list[dict[str, str]]:
    """Rows of the goal ladder table in ``docs/goals.md``."""
    rows, inside = [], False
    for line in text.splitlines():
        if line.startswith("| Id | Goal |"):
            inside = True
            continue
        if not inside:
            continue
        if not line.startswith("|"):
            if rows:
                break
            continue
        if line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
        if len(cells) < 6:
            continue
        rows.append(
            {
                "id": cells[0],
                "goal": cells[1],
                "target": cells[2],
                "kill": cells[3],
                "track": cells[4],
                "status": " | ".join(cells[5:]),
            }
        )
    for r in rows:
        r["verdict"] = verdict_of(r["status"])
        r["category"] = category_of(r["verdict"])
    return rows


def verdict_of(status: str) -> str:
    """The first bold phrase of a status, else its opening words."""
    m = re.search(r"\*\*(.+?)\*\*", status)
    if m and status.index("**") < 40:
        return m.group(1).strip().rstrip(".")
    return re.split(r"[(;.]", status, maxsplit=1)[0].strip()


def category_of(verdict: str) -> str:
    v = verdict.lower()
    if v.startswith("probe"):
        return "probe"
    if v.startswith(("pre-registered", "planned", "in progress", "running")):
        return "pending"
    if "leaning kill" in v or "inconclusive" in v or "not passed" in v:
        return "mixed"
    if v.startswith("kill"):
        return "kill"
    if v.startswith(("passed", "pass", "met", "matched")):
        return "pass"
    return "mixed"


def parse_research_log(text: str) -> dict[str, Any]:
    """``docs/research_log.md`` -> {"intro": str, "tracks": [{"title", "summary",
    "entries": [{"title", <LOG_FIELDS>...}]}]}."""
    intro: list[str] = []
    tracks: list[dict[str, Any]] = []
    entry: dict[str, str] | None = None
    for line in text.splitlines():
        if line.startswith("## Track: "):
            tracks.append(
                {
                    "title": line[len("## Track: ") :].strip(),
                    "summary": "",
                    "entries": [],
                }
            )
            entry = None
        elif line.startswith("### ") and tracks:
            entry = {"title": line[4:].strip()}
            tracks[-1]["entries"].append(entry)
        elif entry is not None and line.startswith("- **"):
            m = re.match(r"- \*\*(.+?):\*\*\s*(.*)", line)
            if m and m.group(1) in LOG_FIELDS:
                entry[m.group(1)] = m.group(2).strip()
        elif tracks and entry is None and line.strip():
            tracks[-1]["summary"] = (tracks[-1]["summary"] + " " + line.strip()).strip()
        elif not tracks and not line.startswith("# "):
            intro.append(line)
    for t in tracks:
        for e in t["entries"]:
            e["category"] = category_of(e.get("Verdict", ""))
    return {"intro": "\n".join(intro).strip(), "tracks": tracks}


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def list_jobs(results: Path = RESULTS) -> list[dict[str, Any]]:
    """Every job folder with a ``status.json``, newest update first."""
    out = []
    for status in (results / "runs").glob("*/status.json"):
        s = read_json(status)
        if not isinstance(s, dict):
            continue
        out.append(
            {
                "id": s.get("id", status.parent.name),
                "state": s.get("state", "?"),
                "lane": (read_json(status.parent / "job.json") or {}).get("lane", "?"),
                "started": s.get("started_at", ""),
                "updated": s.get("updated_at", ""),
                "minutes": s.get("elapsed_min"),
                "dir": str(status.parent),
            }
        )
    return sorted(out, key=lambda j: j["updated"] or "", reverse=True)


def tail(path: Path, n: int = 30) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


# ------------------------------------------------------------------ charts
HOW_TO_READ = {
    "g7_curves": "Each line is the median score of 16 training runs while they learn, per "
    "condition. Higher is better. The goal-aware centres climb; the goal-blind one stays "
    "near −4 because it cannot know which snakes should eat.",
    "g7_heldout": "Average score on the 6 goals no run ever trained on, with 95% "
    "intervals. (d) is the hand-coded reference. The last bar is G7c's 16 fresh runs of "
    "(b): it reaches (d)'s level without the two failed runs.",
    "g12_scatter": "Each dot is one trained squad. Further right = the decoder names the "
    "eaters correctly more often; higher = writing decoded symbols steers a snake better. "
    "Squads that must send the goal in their messages (centre-goal) sit top right; "
    "goal-blind squads have nothing to read.",
    "g2_plant": "Fault-diagnosis quality (macro-F1, higher is better) against how many "
    "bytes each sensor may send. Learned codes stay near 0.97 even at 1 byte, where the "
    "best hand design falls to 0.82.",
    "g13_swarm": "Share of episodes where the leak is found within 2 minutes, by message "
    "size. Dotted lines are silent swarms. Every talking line is flat from 2 bytes: more "
    "bytes do not help, so the radio is not the bottleneck.",
    "multi_probe": "Leaks found within 2 minutes (out of 1 or 3). The gap between "
    "'hand 2B' and 'sensed oracle' is what better coordination could still win. It is "
    "small with 1 leak and large with 3.",
    "r14_brains": "Share of episodes where every leak is found within 2 minutes, by "
    "number of leaks (the brains trained on 3 only). Learned brains sit between the "
    "hand design and the sensed oracle (the coordination bound); with 3 leaks they close "
    "half the gap, and with 4 leaks, which they never saw, they gain the most.",
    "r14_robust": "Share of 3-leak episodes where every leak is found, in harder "
    "conditions than training (neither side retuned). The learned brains beat the hand "
    "design in every one; the gain grows when sensing or positions get worse. With a "
    "wide plume the sensed oracle's fixed rules fail too, so it is no bound there.",
    "r16b_ground": "G14b: the revised station (v5) on a fresh set of 150 sealed "
    "requests. Read it like the G14 chart: green high, red (wrong commands) below the "
    "dashed 3% line.",
    "r16_ground": "The ground station on the 150 sealed requests, by model. Green bars "
    "should be high, the red bar (wrong commands: the swarm flies something nobody "
    "asked) low; the dashed line is the 3% bar. Bigger models are more helpful; the "
    "small ones refuse or ask more and so send fewer wrong commands.",
    "r17_mission": "Person search on 40 held-out missions. Score = share found minus "
    "twice the share of wrong finds (sending the team to the wrong place), in points. "
    "OpenSwarm's default declares the first detection and is misled by false alarms; "
    "the planner that reads the mission comes close to tuning every mission by hand.",
    "r17_plans": "How high the scouts sweep, per held-out mission, against how good the "
    "detector is (px50: the person size in pixels it finds half the time; lower is "
    "better). The mission-reading planner, like per-mission tuning, sweeps higher with a "
    "better detector and over larger areas (hover for the area); the one tuned plan "
    "always sweeps at 4.6 m.",
    "r17_robust": "Person-search score (found minus twice the wrong finds) of each "
    "planner in worse conditions, none retrained. The mission-reading planner keeps its "
    "lead in wind, cold, larger areas and with more drones; every plan that confirms "
    "collapses when the person walks.",
    "r17b_handoff": "Person-search score on missions where the person walks and where "
    "they stand still. The step-1 planner collapses when the person walks; the version 2 "
    "planner (with the hand-off) holds up there but gives up points when the person is "
    "still, because it cannot tell the two apart.",
    "r15b_scout": "What the scout brain (width 0.5, fine-tuned on drone footage) sees "
    "in one frame, by how many pixels wide the person is, at three confidence thresholds. "
    "Rough altitudes for the OV2640 camera are in the labels. At OpenSwarm's 0.7 the "
    "scout misses almost everyone; at 0.5 it finds a third to a half of people up close.",
    "r17e_requirements": "Person found (%) as the scout brain's false detections are "
    "cut, with the operator confirming and plans tuned per mission, for the K210 camera "
    "(224 px) and a 448-px camera. With today's brains no K210 search reaches half the "
    "people; a 10-30 times cut does. The 448-px camera finds 13-22 points more at every "
    "level.",
    "r17f_mission_time": "Person-search score of each way of planning, on missions "
    "where the person may walk (with the operator's hint) and where the person stands "
    "still. Tuning at mission time is the only one good on both.",
    "r17f_budget": "Mission-time planning score against how long the ground station "
    "tunes each mission (one CPU core). Dotted lines: the learned planners. Most of the "
    "gain comes in the first half minute.",
    "r17g_system": "Realistic person searches (walking people, GPS error, wind): the "
    "share found and the share of wrong finds for each camera and false-alarm level, "
    "with plans tuned at mission time. Only the 448-px camera with false alarms cut 10x "
    "finds half the people while keeping wrong finds under 5%.",
    "r17h_closer_look": "The search on R17's 40 held-out missions, each replanned at "
    "mission time: share of people found (green) and of finds that are wrong (red). R19's "
    "confirmer before the operator gives 7.0% wrong finds; adding the closer look at the "
    "hover (R20b's measured cut, its upper bound and its replicate) brings it to 4.7% "
    "(5.6%, 3.7%). Dotted lines: the 40% found and 5% wrong limits.",
    "r17i_wrong_cost": "The search with the closer look, each of R17's 40 held-out "
    "missions retuned at mission time with a stiffer cost per wrong find (the mission "
    "score uses 2). Left: wrong finds; right: people found; 95% CI over missions. Dotted "
    "curves: plans made for the cut's upper bound, flown where the cut is as measured.",
    "r17j_requirement": "The search with the closer look (92% of people kept) on R17's 40 "
    "held-out missions, each tuned at mission time, against the rate at which the evidence "
    "chain accepts an empty candidate. Grey lines: R19's measured rate, the closer look's "
    "and its upper bound. The search meets both limits left of where the red curve crosses "
    "5% and the green stays above 40%.",
    "r17h_sensitivity": "How the recommended search system's share found changes when "
    "one assumption moves. Long red bars are the assumptions to measure first: the "
    "operator wrongly accepting empty views, and how many independent frames a second "
    "of video gives.",
    "r15f_operator": "Left: how often the operator says 'person' on spots that clearly "
    "show a person (green) and on spots that clearly show none (red), for a quick look "
    "(R15e) and a careful one with a wider view (R15f); 95% intervals. Right: the "
    "recommended search system with each operator. A careful look cuts the confirmed "
    "false alarms about threefold, not yet to what the search needs.",
    "r19_confirmer": "Left: on the spots the operator labelled, the share of people each "
    "confirmer keeps (green) and of clearly empty spots it lets through (red), with 95% "
    "intervals; every threshold keeps 95% of val people. Right: the share of VisDrone-"
    "labelled false alarms each model lets through on test-dev (some are unlabelled "
    "people, so this cannot reach zero). The dotted line is the bar: 10% (a 10x cut).",
    "r20_closer_look": "Left: the share of test-dev false alarms (by VisDrone's labels) "
    "that each two-model chain lets through, with 95% intervals; every chain keeps 95% of "
    "val people. The closer look passes 0.86x of the control, where the bar was 0.5x. "
    "Right (R20b): the 594 passed false alarms R20 had not checked, labelled blind; most are "
    "people the labels miss or count twice. On the clearly empty spots (red) the closer look "
    "passes 15 where the control passes 35 (0.43x; replicate 9, 0.26x).",
    "r21_ground_lora": "G14b's 150 sealed requests to the ground station: the share of "
    "requests that should run and came out as exactly the right command (left) and the "
    "share of all requests that became a wrong command (right), with 95% Wilson intervals. "
    "The 14B as is is the deployed G14b model; the 7B runs with the same prompt, as is and "
    "after QLoRA on 997 operator-style requests. Dotted lines: the G14 bars.",
    "r21b_replicate": "G14b's 150 sealed requests to the ground station, run once per "
    "QLoRA adapter of the 7B: seed 0 is R21's, seed 1 is the same recipe with another "
    "training seed. Exact command on the 90 that should run (left) and wrong commands over "
    "all 150 (right), with 95% Wilson intervals; dotted lines are G17's bars.",
    "r21c_fresh_sealed": "G14c: 150 fresh sealed requests (written after the adapters were trained), "
    "run once per system. Exact command on the 90 that should run (left) and wrong commands over all "
    "150 (right), with 95% Wilson intervals; dotted lines are G17's bars, which each adapter must meet.",
    "r22_clean_labels": "R22: 150 random VisDrone test-dev false alarms per scout, labelled "
    "blind, scaled to false alarms per frame. Control: R15b's scouts; cleaned: retrained with "
    "2,456 person boxes an out-of-fold confirmer added. Red is what the bar counts (clearly "
    "not a person); the cleaned scouts' share of real but unlabelled people (blue) grows instead.",
    "r23_k210": "R15b's scout brain as float (blue) and as the int8 model nncase compiles "
    "for OpenSwarm's K210 (orange), run in nncase's simulator on all 64,727 VisDrone "
    "test-dev tiles. Left: the main detection metrics per seed; right: the share of "
    "people found at the 0.5 threshold, by person size in the 224-px tile.",
    "r23c_k210": "People found on VisDrone test-dev by R15b's scout brain as float at 0.5 "
    "(blue), and as the K210 int8 model with its threshold set on val to float's false "
    "detections per tile: R23b's single output (grey) and R23c's split head, where the "
    "heatmap has its own clamped 8-bit output (orange). Left: all people per seed, with the "
    "PASS bar (float minus 2 points) dotted; right: by person size (seed 0).",
    "r24_maixcam": "People found on VisDrone test-dev 448 tiles by R15c's 448-px scout "
    "brain as float at 0.5 (blue), and as the int8 model tpu-mlir compiles for MaixCAM's "
    "cv181x NPU with its threshold set on val to float's false detections per tile: R24 "
    "(grey; heatmap steps of 0.19 logit) and, once run, R24b (orange; the heat conv "
    "calibrated to its clamp, steps of 0.031). Left: all people per seed, with the PASS "
    "bar (float minus 2 points) dotted; right: by person size (seed 0).",
    "r25c_clear": "R25b's four chains on VisDrone test-dev, 400 passed false alarms each, labelled "
    "blind (chain hidden). Left: what those false alarms are (people VisDrone did not label, unclear, "
    "clearly not a person). Right: clear non-persons passed per frame, with the dotted PASS bar at "
    "0.67x the 224 chain.",
    "r25_camera": "VisDrone test-dev at native scale, per frame: the 224-px and 448-px "
    "scout brains (seeds 0 and 1) followed by a confirmer whose threshold keeps 95% of "
    "that scout's val people. Left: labelled false alarms per frame, raised by the scout "
    "(light) and passed by R19's confirmer (dark), and with R25b's confirmer trained on "
    "the 448 scout's own candidates (orange, once run); dotted: the PASS bar, 0.67x the "
    "224 chain. Right: the share of people the whole chain finds.",
    "interlingua": "Score as a share of squad B's own centre. 'raw' = A's centre talks to "
    "B with no translation; 'bridged' = through the decoders; 'oracle' = the best any "
    "translation through the shared meanings could do.",
    "kshot": "Joint accuracy (right service and right fault) on a new system against how "
    "many of its incidents are labelled (k). The adapted specialist climbs far above the "
    "frozen one and the BARO baseline.",
    "r15_scout": "Each point is one trained detector; further right = more memory on "
    "the K210. Higher is better on every line. People who fill part of the frame are "
    "found well (top two lines); small, distant people are the weak spot (bottom line), "
    "and growing the model 8× buys only 6 points there.",
}


def build_charts(res: Path) -> dict[str, Any]:  # pragma: no cover - plotting
    """Chart name -> plotly figure, for the results that exist."""
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    figs: dict[str, Any] = {}
    curves = read_csv(res / "r7b_goal_conditioned_16runs_curves.csv")
    if curves:
        cdf = pd.DataFrame(curves)
        cdf = cdf[cdf["val_mission"] != ""].astype(
            {"generation": int, "val_mission": float}
        )
        med = cdf.groupby(["mode", "generation"])["val_mission"].median().reset_index()
        fig = go.Figure()
        for mode, part in med.groupby("mode"):
            fig.add_trace(
                go.Scatter(x=part["generation"], y=part["val_mission"], name=mode)
            )
        fig.update_layout(
            title="G7: score while training (median of 16 runs)",
            xaxis_title="generation",
            yaxis_title="mission score",
        )
        figs["g7_curves"] = fig
    ref = read_json(res / "r7b_goal_conditioned_16runs.json")
    r7c = read_json(res / "r7c_reliability.json")
    if ref:
        bars = [
            {"condition": c["name"], "held-out score": c["mean"], "ci": c["ci"]}
            for c in ref["sets"]["heldout"]["result"]["conditions"]
        ]
        if r7c:
            kept = [r["heldout"] for r in r7c["per_run"] if r["kept"]]
            n, m = len(kept), sum(kept) / len(kept)
            sd = (sum((x - m) ** 2 for x in kept) / (n - 1)) ** 0.5
            bars.append({"condition": "(b) fresh runs, G7c", "held-out score": m,
                         "ci": 2.13 * sd / n**0.5})  # fmt: skip
        figs["g7_heldout"] = px.bar(
            pd.DataFrame(bars), x="condition", y="held-out score", error_y="ci",
            title="G7/G7c: score on goals never trained on",
        )  # fmt: skip
    r7d = read_json(res / "r7d_legibility.json")
    if r7d:
        pts = [
            {"condition": r["mode"], "run": r["run"],
             "eaters read correctly": r["read"]["eaters_exact"],
             "write efficacy": r["write"]["efficacy"]}
            for r in r7d["runs"]
            if r["write"]["efficacy"] == r["write"]["efficacy"]
        ]  # fmt: skip
        figs["g12_scatter"] = px.scatter(
            pd.DataFrame(pts), x="eaters read correctly", y="write efficacy",
            color="condition", hover_data=["run"],
            title="G12: every squad's language, read and written",
        )  # fmt: skip
    plant = read_json(res / "plant_learned_codes.json")
    if plant:
        pdf = pd.DataFrame([r for r in plant["rows"] if r.get("variant") == "base"])
        line = pdf.groupby(["encoding", "budget"])["macro_f1"].mean().reset_index()
        hand = {t["budget"]: t["hand_mean"] for t in plant["g2_tests"]}
        line = pd.concat(
            [line, pd.DataFrame([{"encoding": "best hand design", "budget": b,
                                  "macro_f1": v} for b, v in hand.items()])]
        )  # fmt: skip
        figs["g2_plant"] = px.line(
            line, x="budget", y="macro_f1", color="encoding", markers=True, log_x=True,
            title="G2: diagnosis quality by bytes per sensor message",
        )  # fmt: skip
    swarm = read_json(res / "r14_swarm_talk.json")
    if swarm:
        pts = [
            {"budget": c["budget"], "loss": c["loss"], "drift": c["drift"],
             "found": sum(c["success"]) / len(c["success"])}
            for c in swarm["cells"].values()
        ]  # fmt: skip
        sdf = pd.DataFrame(pts)
        talk = sdf[sdf["budget"] > 0].copy()
        talk["setting"] = (
            "loss " + talk["loss"].astype(str) + ", drift " + talk["drift"].astype(str)
        )
        fig = px.line(
            talk, x="budget", y="found", color="setting", markers=True, log_x=True,
            title="G13: leak found within 2 min, by bytes per message",
        )  # fmt: skip
        for _, r in sdf[sdf["budget"] == 0].iterrows():
            fig.add_hline(y=r["found"], line_dash="dot",
                          annotation_text=f"silent, drift {r['drift']}")  # fmt: skip
        figs["g13_swarm"] = fig
    multi = read_json(res / "r14_multi_probe.json")
    if multi:
        pts = [
            {"leaks": key.split("|")[0], "system": key.split("|")[1],
             "found": v["found"][0], "ci": v["found"][1], "all found": v["all"][0]}
            for key, v in multi["summary"].items()
            if key.count("|") == 1
        ]  # fmt: skip
        figs["multi_probe"] = px.bar(
            pd.DataFrame(pts), x="system", y="found", error_y="ci", color="leaks",
            barmode="group", hover_data=["all found"],
            title="Probe: leaks found in 2 min, one leak vs three",
        )  # fmt: skip
    brains = read_json(res / "r14_brains.json")
    if brains:
        names = {"silent": "silent", "hand 2B": "hand design",
                 "sensed oracle": "sensed oracle"}  # fmt: skip
        pts = []
        for k, r in brains["by_k"].items():
            pts.append({"leaks": str(k), "system": "learned brains",
                        "all found (%)": 100 * r["learned"]["all"][0],
                        "ci": 100 * r["learned"]["all"][1]})  # fmt: skip
            for key, name in names.items():
                pts.append({"leaks": str(k), "system": name,
                            "all found (%)": 100 * r["refs"][key]["all"], "ci": 0.0})  # fmt: skip
        order = ["silent", "hand design", "learned brains", "sensed oracle"]
        figs["r14_brains"] = px.bar(
            pd.DataFrame(pts), x="leaks", y="all found (%)", error_y="ci",
            color="system", barmode="group", category_orders={"system": order},
            title="R14 step 2: learned per-role brains vs the hand design",
        )  # fmt: skip
    robust = read_json(res / "r14_robust.json") or read_json(
        res / "r14_robust_step2.json"
    )
    if robust:
        names = {"hand 2B": "hand design", "step2": "brains (trained at nominal)",
                 "random": "brains (randomised training)",
                 "sensed oracle": "sensed oracle"}  # fmt: skip
        pts = [
            {"condition": c, "system": name,
             "all found (%)": 100 * robust["by_condition"][c][key]["all"]}
            for c in robust["conditions"]
            for key, name in names.items()
            if key in robust["by_condition"][c]
        ]  # fmt: skip
        figs["r14_robust"] = px.bar(
            pd.DataFrame(pts), x="condition", y="all found (%)", color="system",
            barmode="group", category_orders={"system": list(names.values())},
            title="R14 step 3: the brains in harder conditions (3 leaks)",
        )  # fmt: skip
    measures = {"exact": "exact command", "asked_when_unclear": "asked when unclear",
                "refused_when_should": "refused when should",
                "wrong_commands": "wrong command"}  # fmt: skip
    for chart, stem, goal in (("r16_ground", "r16_ground_station", "G14"),
                              ("r16b_ground", "r16b_ground_station", "G14b")):  # fmt: skip
        ground = read_json(res / f"{stem}.json")
        if not ground:
            continue
        pts = [
            {"translator": r["translator"].replace("ollama:", ""), "measure": label,
             "%": 100 * r[key][0]}
            for name, r in ground["runs"].items()
            if name.split("|")[0] in ("sealed", "sealed-b")
            for key, label in measures.items()
        ]  # fmt: skip
        fig = px.bar(
            pd.DataFrame(pts), x="translator", y="%", color="measure", barmode="group",
            color_discrete_sequence=["#2e7d32", "#66bb6a", "#a5d6a7", "#c62828"],
            title=f"{goal}: the ground station on its sealed requests",
        )  # fmt: skip
        fig.add_hline(y=3, line_dash="dash", line_color="#c62828")
        figs[chart] = fig
    mission = read_json(res / "r17_mission.json")
    if mission:
        pts = [
            {"system": name, "measure": label, "points": 100 * sys[key]}
            for name, sys in mission["systems"].items()
            for key, label in (("score", "score"), ("found", "found"),
                               ("wrong", "wrong find"))
        ]  # fmt: skip
        figs["r17_mission"] = px.bar(
            pd.DataFrame(pts), x="system", y="points", color="measure",
            barmode="group", title="R17: planning the person search",
        )  # fmt: skip
        pts = [
            {"px50 (lower = better detector)": row["px50"], "system": name,
             "sweep altitude (m)": plan["sweep_alt"],
             "area (m²)": row["width"] * row["height"]}
            for row in mission.get("plans", [])
            for name, plan in row["systems"].items()
            if name in ("blind", "conditioned", "per-mission tuned")
        ]  # fmt: skip
        if pts:
            figs["r17_plans"] = px.scatter(
                pd.DataFrame(pts), x="px50 (lower = better detector)",
                y="sweep altitude (m)", color="system", hover_data=["area (m²)"],
                title="R17: the sweep altitude each planner chooses",
            )  # fmt: skip
    robust17 = read_json(res / "r17_robust.json")
    if robust17:
        pts = [
            {"condition": c, "system": name, "score (points)": 100 * row[name]["score"]}
            for c, row in robust17["by_condition"].items()
            for name in ("OpenSwarm default", "hand two-tier", "blind", "conditioned")
        ]  # fmt: skip
        figs["r17_robust"] = px.bar(
            pd.DataFrame(pts), x="condition", y="score (points)", color="system",
            barmode="group", title="R17 step 1b: the search planners in a worse world",
        )  # fmt: skip
    handoff = read_json(res / "r17b_handoff.json")
    if handoff:
        pts = [
            {"system": name, "person": person, "score (points)": 100 * row[person]}
            for name, row in handoff["v2"].items()
            for person in ("walking", "still")
        ]  # fmt: skip
        figs["r17b_handoff"] = px.bar(
            pd.DataFrame(pts), x="system", y="score (points)", color="person",
            barmode="group", title="R17 step 2: walking and still people",
        )  # fmt: skip
    aerial = read_json(res / "r15b_aerial_scout.json")
    if aerial:
        runs = [r for r in aerial["runs"]
                if r["condition"] == "fine-tuned" and r["alpha"] == 0.5]  # fmt: skip
        bins = {"0-8": "0-8 px (above ~8 m)", "8-16": "8-16 px (~4-8 m)",
                "16-32": "16-32 px (~2-4 m)", "32+": "32+ px (below ~2 m)"}  # fmt: skip
        pts = [
            {"person size": label, "threshold": thr,
             "recall (%)": 100 * sum(
                 r["test"]["pooled"]["operating"][thr][f"recall {b}"] for r in runs)
             / len(runs)}
            for thr in ("0.3", "0.5", "0.7") for b, label in bins.items()
            if runs and thr in runs[0]["test"]["pooled"]["operating"]
        ]  # fmt: skip
        if pts:
            figs["r15b_scout"] = px.line(
                pd.DataFrame(pts), x="person size", y="recall (%)", color="threshold",
                markers=True, title="R15b: what the scout sees per frame",
            )  # fmt: skip
    req = read_json(res / "r17e_requirements.json")
    if req:
        pts = [
            {"false detections (x measured)": r["fa_mult"],
             "found (%)": 100 * r[system]["found"],
             "plan": f"{r['camera']}: {system}, {r['frames']:g} frame(s)/look",
             "wrong finds (%)": 100 * r[system]["wrong"]}
            for r in req["rows"] for system in ("tuned per mission",)
        ]  # fmt: skip
        fig = px.line(
            pd.DataFrame(pts), x="false detections (x measured)", y="found (%)",
            color="plan", markers=True, log_x=True, hover_data=["wrong finds (%)"],
            title="R17 step 4: how far the scout's false alarms must fall",
        )  # fmt: skip
        fig.update_xaxes(autorange="reversed")
        fig.add_hline(y=50, line_dash="dot")
        figs["r17e_requirements"] = fig
    mt = read_json(res / "r17f_mission_time.json")
    if mt:
        pts = [
            {"planner": name, "missions": which, "score (points)": 100 * v}
            for which, row in mt["table"].items() for name, v in row.items()
        ]  # fmt: skip
        figs["r17f_mission_time"] = px.bar(
            pd.DataFrame(pts), x="planner", y="score (points)", color="missions",
            barmode="group", title="R17 step 5: plan at mission time",
        )  # fmt: skip
    budget = read_json(res / "r17f_budget.json")
    if budget:
        pts = [
            {"seconds of tuning (one core)": p["seconds_median"], "missions": which,
             "score (points)": 100 * p[which]}
            for p in budget["points"]
            for which in ("hint 85% right", "still person, hint still")
        ]  # fmt: skip
        fig = px.line(
            pd.DataFrame(pts), x="seconds of tuning (one core)", y="score (points)",
            color="missions", markers=True, log_x=True,
            title="R17 step 5b: how much tuning time a mission is worth",
        )  # fmt: skip
        for name, ref in budget["references"].items():
            for which, v in ref.items():
                fig.add_hline(y=100 * v, line_dash="dot",
                              annotation_text=f"{name} ({which})")  # fmt: skip
        figs["r17f_budget"] = fig
    system = read_json(res / "r17g_system.json")
    if system:
        pts = [
            {"setup": f"{r['camera']}, false x{r['fa_mult']:g}, {r['frames']:g} f/look",
             "measure": label, "%": 100 * r["mission-time tuning"][key]}
            for r in system["rows"]
            for key, label in (("found", "found"), ("wrong", "wrong finds"))
        ]  # fmt: skip
        figs["r17g_system"] = px.bar(
            pd.DataFrame(pts), x="setup", y="%", color="measure", barmode="group",
            color_discrete_sequence=["#2e7d32", "#c62828"],
            title="R17 step 6: realistic searches with mission-time planning",
        )  # fmt: skip
    sens = read_json(res / "r17h_sensitivity.json")
    if sens:
        rows = sorted(sens["rows"], key=lambda r: r["found_change"])
        fig = px.bar(
            pd.DataFrame([{"assumption": r["variant"],
                           "change in found (points)": 100 * r["found_change"],
                           "wrong finds (%)": 100 * r["wrong"]} for r in rows]),
            x="change in found (points)", y="assumption", orientation="h",
            color="change in found (points)", color_continuous_scale="RdYlGn",
            hover_data=["wrong finds (%)"],
            title="R17 step 6b: which assumption matters most",
        )  # fmt: skip
        figs["r17h_sensitivity"] = fig
    closer_search = read_json(res / "r17h_closer_look.json")
    if closer_search:
        rows = sorted(closer_search["rows"], key=lambda r: -r["wrong"])
        fig = go.Figure()
        for key, label, colour in (("found", "found", "#2e7d32"),
                                   ("wrong", "wrong finds", "#c62828")):  # fmt: skip
            fig.add_trace(go.Bar(x=[r["variant"] for r in rows], y=[100 * r[key] for r in rows],
                                 name=label, marker_color=colour))  # fmt: skip
        fig.add_hline(y=5, line_dash="dot", line_color="#c62828", annotation_text="5% wrong")
        fig.add_hline(y=40, line_dash="dot", line_color="#2e7d32", annotation_text="40% found")
        fig.update_layout(barmode="group", yaxis_title="%",
                          title="R17 step 6f: the search with the closer look")  # fmt: skip
        figs["r17h_closer_look"] = fig
    cost = read_json(res / "r17i_wrong_cost.json")
    if cost:
        from plotly.subplots import make_subplots

        short = {"R20b: confirmer and closer look at the hover, then operator (92% / 2.1%)":
                 "measured cut", "R20b: the same at the replicate's cut (92% / 1.3%)":
                 "replicate's cut", "R20b: the same at the cut's upper bound (92% / 3.2%)":
                 "upper bound", "R19: confirmer at the ground, then operator (92% / 5%)":
                 "R19 alone"}  # fmt: skip
        fig = make_subplots(rows=1, cols=2, subplot_titles=("wrong finds", "people found"))
        lines: dict[str, list] = {}
        for r in cost["rows"]:
            name = short[r["plan"]] if r["plan"] == r["world"] else (
                f"planned for {short[r['plan']]}, flown at {short[r['world']]}")  # fmt: skip
            lines.setdefault(name, []).append(r)
        for k, (name, rs) in enumerate(lines.items()):
            colour = px.colors.qualitative.Safe[k % 10]
            dash = "solid" if " flown at " not in name else "dot"
            for col, key in ((1, "wrong"), (2, "found")):
                fig.add_trace(go.Scatter(
                    x=[r["cost"] for r in rs], y=[100 * r[key] for r in rs], name=name,
                    mode="lines+markers", line={"color": colour, "dash": dash},
                    showlegend=col == 1, legendgroup=name,
                    error_y={"type": "data", "array": [100 * r[key + "_hw"] for r in rs]}),
                    1, col)  # fmt: skip
        fig.add_hline(y=5, line_dash="dot", line_color="#c62828", row=1, col=1,
                      annotation_text="5% limit")  # fmt: skip
        fig.add_hline(y=40, line_dash="dot", line_color="#2e7d32", row=1, col=2,
                      annotation_text="40% limit")  # fmt: skip
        fig.update_xaxes(type="log", title_text="cost of a wrong find in the tuning objective")
        fig.update_layout(yaxis_title="%", title=(
            f"R17 step 6g: plan to the wrong-find limit ({cost['verdict']})"))  # fmt: skip
        figs["r17i_wrong_cost"] = fig
    need = read_json(res / "r17j_requirement.json")
    if need:
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=2, subplot_titles=("wrong finds", "people found"))
        for k, c in enumerate(sorted({r["cost"] for r in need["rows"]})):
            rs = [r for r in need["rows"] if r["cost"] == c]
            for col, key in ((1, "wrong"), (2, "found")):
                fig.add_trace(go.Scatter(
                    x=[100 * r["op_false"] for r in rs], y=[100 * r[key] for r in rs],
                    name=f"wrong-find cost {c:g}", mode="lines+markers", legendgroup=str(c),
                    showlegend=col == 1, line={"color": px.colors.qualitative.Safe[k]},
                    error_y={"type": "data", "array": [100 * r[key + "_hw"] for r in rs]}),
                    1, col)  # fmt: skip
        fig.add_hline(y=5, line_dash="dot", line_color="#c62828", row=1, col=1,
                      annotation_text="5% limit")  # fmt: skip
        fig.add_hline(y=40, line_dash="dot", line_color="#2e7d32", row=1, col=2,
                      annotation_text="40% limit")  # fmt: skip
        for x, label in ((4.9, "R19"), (2.1, "closer look"), (3.2, "its upper bound")):
            fig.add_vline(x=x, line_dash="dash", line_color="#9e9e9e", annotation_text=label)
        fig.update_xaxes(title_text="false accepts per empty candidate (%)")
        fig.update_layout(yaxis_title="%",
                          title="R17 step 6h: the false-accept rate the evidence must reach")  # fmt: skip
        figs["r17j_requirement"] = fig
    quick, careful = read_json(res / "r15e_operator.json"), read_json(
        res / "r15f_careful.json"
    )
    search = read_json(res / "r17h_careful.json")
    if quick and careful and search:
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            "operator's answers (clear cases)", "the search with that operator"))  # fmt: skip
        looks = {
            "quick look (R15e)": quick["rows"][0],
            "careful look (R15f)": careful["rows"][0],
        }
        for key, label, colour in (("hit", "says person: a person", "#2e7d32"),
                                   ("false_accept", "says person: empty spot", "#c62828")):  # fmt: skip
            vals = [r[key] for r in looks.values()]
            fig.add_trace(go.Bar(
                x=list(looks), y=[100 * v[0] for v in vals], name=label, marker_color=colour,
                error_y={"type": "data", "array": [100 * (v[2] - v[0]) for v in vals],
                         "arrayminus": [100 * (v[0] - v[1]) for v in vals]}), 1, 1)  # fmt: skip
        names = {"recommended (as in step 6)": "assumed (90% / 2%)",
                 "measured operator, clear cases (92% / 23%)": "quick look",
                 "careful operator, clear cases (90% / 8%)": "careful look"}  # fmt: skip
        rows = [r for r in search["rows"] if r["variant"] in names]
        for key, label, colour in (("found", "found", "#2e7d32"),
                                   ("wrong", "wrong finds", "#c62828")):  # fmt: skip
            fig.add_trace(go.Bar(x=[names[r["variant"]] for r in rows],
                                 y=[100 * r[key] for r in rows], name=label,
                                 marker_color=colour, opacity=0.6), 1, 2)  # fmt: skip
        fig.update_layout(barmode="group", yaxis_title="%",
                          title="R15f: a careful look by the operator, and the search")  # fmt: skip
        figs["r15f_operator"] = fig
    conf = read_json(res / "r19_confirmer.json")
    if conf:
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            "the 400 operator spots (clear cases)", "test-dev, false alarms passed"))  # fmt: skip
        whos = {"operator alone": conf["operator"]}
        chosen = next((s for s in conf["systems"]
                       if f"{s['model']}_s{s['seed']}" == conf["chosen"]), None)  # fmt: skip
        for s in conf["systems"]:  # the chosen seed for its model, seed 0 for the others
            if "spots" in s and (s is chosen or (s["seed"] == 0 and not (
                    chosen and s["model"] == chosen["model"]))):  # fmt: skip
                whos[s["name"]] = s["spots"]
        if chosen and "chain" in chosen:
            whos[f"{chosen['name']} + operator"] = chosen["chain"]
        for key, label, colour in (("matched", "people kept", "#2e7d32"),
                                   ("not_person", "non-persons passed", "#c62828")):  # fmt: skip
            vals = [v[key] for v in whos.values()]
            fig.add_trace(go.Bar(
                x=list(whos), y=[100 * v[0] for v in vals], name=label, marker_color=colour,
                error_y={"type": "data", "array": [100 * (v[2] - v[0]) for v in vals],
                         "arrayminus": [100 * (v[0] - v[1]) for v in vals]}), 1, 1)  # fmt: skip
        tests = [s for s in conf["systems"] if "test" in s]
        fig.add_trace(go.Bar(
            x=[f"{s['name']} (seed {s['seed']})" for s in tests],
            y=[100 * s["test"]["pass"][0] for s in tests], name="false alarms passed",
            marker_color="#c62828", opacity=0.6, showlegend=False,
            error_y={"type": "data", "array": [100 * (s["test"]["pass"][2] - s["test"]["pass"][0]) for s in tests],
                     "arrayminus": [100 * (s["test"]["pass"][0] - s["test"]["pass"][1]) for s in tests]}),
            1, 2)  # fmt: skip
        fig.add_hline(y=10, line_dash="dot", line_color="#555", row=1, col=1,
                      annotation_text="bar: 10%")  # fmt: skip
        fig.update_layout(barmode="group", yaxis_title="%", title=(
            f"R19: an open-weight confirmer before the operator ({conf['verdict']})"))  # fmt: skip
        figs["r19_confirmer"] = fig
    closer = read_json(res / "r20_closer_look.json")
    if closer and "verdict" in closer:
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            "test-dev false alarms passed (VisDrone labels)",
            "R20b: what each chain passes among 594 false alarms checked blind"))  # fmt: skip
        names = {"control": "scout's view, two models", "closer": "scout's view + closer look",
                 "closer_replicate": "the same, other seeds"}  # fmt: skip
        vals = [closer["chains"][k]["test_pass"] for k in names]
        fig.add_trace(go.Bar(
            x=list(names.values()), y=[100 * v[0] for v in vals], marker_color="#c62828",
            name="false alarms passed", showlegend=False,
            error_y={"type": "data", "array": [100 * (v[2] - v[0]) for v in vals],
                     "arrayminus": [100 * (v[0] - v[1]) for v in vals]}), 1, 1)  # fmt: skip
        vis = closer.get("visual_b")
        if vis:  # R20b: what each chain passes among the 594 spots checked blind
            for key, label, colour in (("person", "a person", "#2e7d32"),
                                       ("ambiguous", "unclear", "#9e9e9e"),
                                       ("not_person", "clearly not a person", "#c62828")):  # fmt: skip
                fig.add_trace(go.Bar(x=list(names.values()), y=[vis[k][key] for k in names],
                                     name=label, marker_color=colour,
                                     text=[vis[k][key] for k in names]), 1, 2)  # fmt: skip
        fig.update_layout(barmode="stack", yaxis_title="%", title=(
            f"R20: a closer look against a second model on the scout's view "
            f"({closer['verdict']}, {closer['ratio']['closer'][0]:.2f}x)"))  # fmt: skip
        fig.update_yaxes(title_text="spots", row=1, col=2)
        figs["r20_closer_look"] = fig
    lora = read_json(res / "r21_ground_lora.json")
    if lora and "verdict" in lora:
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            "exact command (requests that should run)", "wrong commands (all requests)"))  # fmt: skip
        whos = {}
        if lora.get("g14b"):
            whos["14B as is (G14b)"] = lora["g14b"]["g14"]
        names = {"asis_sealed-b": "7B as is", "lora_sealed-b": "7B + QLoRA"}
        whos.update({v: lora["systems"][k] for k, v in names.items() if k in lora["systems"]})
        for col, key, colour, bar in ((1, "exact", "#2e7d32", 90), (2, "wrong_commands", "#c62828", 3)):
            vals = [w[key] for w in whos.values()]
            fig.add_trace(go.Bar(
                x=list(whos), y=[100 * v[0] for v in vals], marker_color=colour,
                showlegend=False, text=[f"{100 * v[0]:.1f}%" for v in vals],
                error_y={"type": "data", "array": [100 * (v[2] - v[0]) for v in vals],
                         "arrayminus": [100 * (v[0] - v[1]) for v in vals]}), 1, col)  # fmt: skip
            fig.add_hline(y=bar, line_dash="dot", line_color="#555", row=1, col=col,
                          annotation_text=f"bar {bar}%")  # fmt: skip
        fig.update_layout(yaxis_title="%", title=(
            f"R21: the ground station's LLM with its weights edited ({lora['verdict']})"))  # fmt: skip
        figs["r21_ground_lora"] = fig
    rep_b = read_json(res / "r21b_replicate.json")
    if rep_b and "verdict" in rep_b:
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            "exact command (requests that should run)", "wrong commands (all requests)"))  # fmt: skip
        whos = {"seed 0 (R21)": rep_b["seed0"], "seed 1": rep_b["seed1"]}
        for col, key, colour, bar in ((1, "exact", "#2e7d32", 90), (2, "wrong_commands", "#c62828", 3)):
            vals = [w[key] for w in whos.values()]
            fig.add_trace(go.Bar(
                x=list(whos), y=[100 * v[0] for v in vals], marker_color=colour,
                showlegend=False, text=[f"{100 * v[0]:.1f}%" for v in vals],
                error_y={"type": "data", "array": [100 * (v[2] - v[0]) for v in vals],
                         "arrayminus": [100 * (v[0] - v[1]) for v in vals]}), 1, col)  # fmt: skip
            fig.add_hline(y=bar, line_dash="dot", line_color="#555", row=1, col=col,
                          annotation_text=f"bar {bar}%")  # fmt: skip
        fig.update_layout(yaxis_title="%", title=(
            f"R21b: the ground station's QLoRA edit, second seed ({rep_b['verdict']})"))  # fmt: skip
        figs["r21b_replicate"] = fig
    fresh = read_json(res / "r21c_fresh_sealed.json")
    if fresh and "verdict" in fresh:
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            "exact command (requests that should run)", "wrong commands (all requests)"))  # fmt: skip
        names = {"lora": "7B + QLoRA, seed 0", "lora_s1": "7B + QLoRA, seed 1",
                 "asis": "7B as is", "q14": "14B as is"}  # fmt: skip
        whos = {names[k]: v for k, v in fresh["systems"].items()}
        for col, key, colour, bar in ((1, "exact", "#2e7d32", 90), (2, "wrong_commands", "#c62828", 3)):
            vals = [w[key] for w in whos.values()]
            fig.add_trace(go.Bar(
                x=list(whos), y=[100 * v[0] for v in vals], marker_color=colour,
                showlegend=False, text=[f"{100 * v[0]:.1f}%" for v in vals],
                error_y={"type": "data", "array": [100 * (v[2] - v[0]) for v in vals],
                         "arrayminus": [100 * (v[0] - v[1]) for v in vals]}), 1, col)  # fmt: skip
            fig.add_hline(y=bar, line_dash="dot", line_color="#555", row=1, col=col,
                          annotation_text=f"bar {bar}%")  # fmt: skip
        fig.update_layout(yaxis_title="%", title=(
            f"R21c: the ground station on a fresh sealed set ({fresh['verdict']})"))  # fmt: skip
        figs["r21c_fresh_sealed"] = fig
    clean = read_json(res / "r22_clean_labels.json")
    if clean and "sample_counts" in clean:
        scouts = list(clean["sample_counts"])
        fig = go.Figure()
        for cls, label, colour in (("person", "unlabelled person", "#1565c0"),
                                   ("ambiguous", "unclear", "#9e9e9e"),
                                   ("not_person", "clearly not a person", "#c62828")):  # fmt: skip
            fig.add_trace(go.Bar(
                x=scouts, name=label, marker_color=colour,
                y=[clean["models"][m]["test_fa_per_frame"] * clean["sample_counts"][m][cls]
                   / sum(clean["sample_counts"][m].values()) for m in scouts]))  # fmt: skip
        fig.update_layout(barmode="stack", yaxis_title="false alarms per test-dev frame",
                          title=f"R22: scouts trained on cleaned labels ({clean['verdict']}, "
                                f"{clean['ratio']['pooled'][0]:.2f}x clear non-persons)")  # fmt: skip
        figs["r22_clean_labels"] = fig
    k210 = read_json(res / "r23_k210.json")
    if k210 and "verdict" in k210:
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            "test-dev, float against the K210 model", "recall per person size at 0.5 (seed 0)"))  # fmt: skip
        tags = list(k210["models"])
        for key, label in (("centre_ap", "centre AP"), ("ap50", "AP50"),
                           ("present_auc", "present AUC")):  # fmt: skip
            for kind, colour in (("float", "#1565c0"), ("int8", "#ef6c00")):
                fig.add_trace(go.Bar(
                    x=[f"{label}, seed {t[-1]}" for t in tags],
                    y=[k210["models"][t][key][kind] for t in tags], name=kind,
                    marker_color=colour, legendgroup=kind, showlegend=key == "centre_ap"),
                    1, 1)  # fmt: skip
        row = k210["models"][tags[0]]["operating"]["0.5"]
        bins = ("0-8", "8-16", "16-32", "32+")
        for kind, colour in (("float", "#1565c0"), ("int8", "#ef6c00")):
            fig.add_trace(go.Bar(x=[f"{b} px" for b in bins],
                                 y=[100 * row[f"recall {b}"][kind] for b in bins],
                                 marker_color=colour, showlegend=False), 1, 2)  # fmt: skip
        fig.update_yaxes(title_text="%", row=1, col=2)
        fig.update_layout(barmode="group", title=(
            f"R23: the scout brain through the K210 toolchain ({k210['verdict']})"))  # fmt: skip
        figs["r23_k210"] = fig
    k210b, k210c = read_json(res / "r23b_k210_threshold.json"), read_json(res / "r23c_k210_fixes.json")
    if k210b and k210c and "verdict" in k210c:
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            "people found, all sizes", "people found by size (seed 0)"))  # fmt: skip
        seeds = ("seed0", "seed1")
        rows = {s_: (k210b["models"][f"r15b_coco_a0.5_{s_}"]["test"],
                     k210c["models"][f"r15b_coco_a0.5_{s_}__heat"]["test"]) for s_ in seeds}  # fmt: skip
        bins = ("all", "0-8", "8-16", "16-32", "32+")
        for label, colour, pick in (("float", "#1565c0", lambda b, c, k: c[k]["float"]),
                                    ("R23b int8", "#9e9e9e", lambda b, c, k: b[k]["int8"]),
                                    ("R23c int8", "#ef6c00", lambda b, c, k: c[k]["int8"])):  # fmt: skip
            fig.add_trace(go.Bar(x=[f"seed {s_[-1]}" for s_ in seeds],
                                 y=[100 * pick(*rows[s_], "recall all") for s_ in seeds],
                                 name=label, marker_color=colour, legendgroup=label), 1, 1)  # fmt: skip
            fig.add_trace(go.Bar(x=[f"{b} px" for b in bins[1:]],
                                 y=[100 * pick(*rows["seed0"], f"recall {b}") for b in bins[1:]],
                                 marker_color=colour, legendgroup=label, showlegend=False), 1, 2)  # fmt: skip
        for i, s_ in enumerate(seeds):
            bar = 100 * rows[s_][1]["recall all"]["float"] - 2
            fig.add_shape(type="line", x0=i - 0.45, x1=i + 0.45, y0=bar, y1=bar,
                          line={"dash": "dot", "color": "#555"}, row=1, col=1)  # fmt: skip
        fig.update_yaxes(title_text="%", row=1, col=1)
        fig.update_layout(barmode="group", title=(
            f"R23c: the K210 scout with the heatmap in its own output ({k210c['verdict']})"))  # fmt: skip
        figs["r23c_k210"] = fig
    mx, mxb = read_json(res / "r24_maixcam.json"), read_json(res / "r24b_maixcam_heatcal.json")
    if mx and "verdict" in mx:
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            "people found, all sizes", "people found by size (seed 0)"))  # fmt: skip
        tags = list(mx["models"])
        bins = ("0-8", "8-16", "16-32", "32+")
        series = [("float", "#1565c0", mx, "float"), ("R24 int8", "#9e9e9e", mx, "int8")]
        if mxb and "verdict" in mxb:
            series.append(("R24b int8", "#ef6c00", mxb, "int8"))
        for label, colour, rep, kind in series:
            fig.add_trace(go.Bar(x=[f"seed {t.split('_seed')[1][0]}" for t in tags],
                                 y=[100 * rep["models"][t]["test"]["recall all"][kind] for t in tags],
                                 name=label, marker_color=colour, legendgroup=label), 1, 1)  # fmt: skip
            fig.add_trace(go.Bar(x=[f"{b} px" for b in bins],
                                 y=[100 * rep["models"][tags[0]]["test"][f"recall {b}"][kind]
                                    for b in bins],
                                 marker_color=colour, legendgroup=label, showlegend=False), 1, 2)  # fmt: skip
        for i, t in enumerate(tags):
            bar = 100 * mx["models"][t]["test"]["recall all"]["float"] - 2
            fig.add_shape(type="line", x0=i - 0.45, x1=i + 0.45, y0=bar, y1=bar,
                          line={"dash": "dot", "color": "#555"}, row=1, col=1)  # fmt: skip
        fig.update_yaxes(title_text="%", row=1, col=1)
        verdicts = f"R24 {mx['verdict']}" + (f", R24b {mxb['verdict']}" if mxb else "")
        fig.update_layout(barmode="group", title=(
            f"R24: the 448-px scout brain through MaixCAM's toolchain ({verdicts})"))  # fmt: skip
        figs["r24_maixcam"] = fig
    cam, camb = read_json(res / "r25_camera.json"), read_json(res / "r25b_camera_own.json")
    if cam and "verdict" in cam:
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            "false alarms per frame", "people found by the chain"))  # fmt: skip
        xs = [f"seed {sd[-1]}, {k} px" for sd in cam["seeds"] for k in ("224", "448")]
        rows = [cam["seeds"][sd][k] for sd in cam["seeds"] for k in ("224", "448")]
        fig.add_trace(go.Bar(x=xs, y=[r["false_candidates_per_frame"] for r in rows],
                             name="raised by the scout", marker_color="#90a4ae"), 1, 1)  # fmt: skip
        fig.add_trace(go.Bar(x=xs, y=[r["false_per_frame"] for r in rows],
                             name="passed by R19's confirmer", marker_color="#37474f"), 1, 1)  # fmt: skip
        fig.add_trace(go.Bar(x=xs, y=[100 * r["people_found"] for r in rows],
                             marker_color="#37474f", showlegend=False), 1, 2)  # fmt: skip
        if camb and "verdict" in camb:
            own = {f"seed {sd[-1]}, 448 px": camb["seeds"][sd]["448"] for sd in camb["seeds"]}
            fig.add_trace(go.Bar(x=list(own), y=[r["false_per_frame"] for r in own.values()],
                                 name="passed by its own confirmer (R25b)",
                                 marker_color="#ef6c00"), 1, 1)  # fmt: skip
            fig.add_trace(go.Bar(x=list(own), y=[100 * r["people_found"] for r in own.values()],
                                 marker_color="#ef6c00", showlegend=False), 1, 2)  # fmt: skip
        for i, sd in enumerate(cam["seeds"]):
            bar = 0.67 * cam["seeds"][sd]["224"]["false_per_frame"]
            fig.add_shape(type="line", x0=2 * i + 0.55, x1=2 * i + 1.45, y0=bar, y1=bar,
                          line={"dash": "dot", "color": "#555"}, row=1, col=1)  # fmt: skip
        fig.update_yaxes(title_text="per frame", row=1, col=1)
        fig.update_yaxes(title_text="%", row=1, col=2)
        verdicts = f"R25 {cam['verdict']}" + (f", R25b {camb['verdict']}" if camb else "")
        fig.update_layout(barmode="group", title=(
            f"R25: the 448-px camera through the confirmer ({verdicts})"))  # fmt: skip
        figs["r25_camera"] = fig
    clear = read_json(res / "r25c_camera_clear.json")
    if clear and "verdict" in clear:
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            "what the passed false alarms are", "clear non-persons passed per frame"))  # fmt: skip
        names = {t: t.replace("s224_seed", "224 px, seed ").replace("s448_seed", "448 px, seed ")
                 for t in clear["chains"]}  # fmt: skip
        xs = [names[t] for t in clear["chains"]]
        rows = list(clear["chains"].values())
        for key, label, color in (("person", "unlabelled person", "#90a4ae"),
                                  ("ambiguous", "unclear", "#cfd8dc"),
                                  ("not_person", "clearly not a person", "#c62828")):  # fmt: skip
            fig.add_trace(go.Bar(x=xs, y=[100 * r[key] / r["sampled"] for r in rows], name=label,
                                 marker_color=color), 1, 1)  # fmt: skip
        fig.add_trace(go.Bar(x=xs, y=[r["clear_per_frame"] for r in rows], showlegend=False,
                             marker_color=["#37474f" if "224" in x else "#ef6c00" for x in xs]),
                      1, 2)  # fmt: skip
        for i, a in enumerate(t for t in clear["chains"] if "224" in t):
            bar = 0.67 * clear["chains"][a]["clear_per_frame"]
            fig.add_shape(type="line", x0=2 * i + 0.55, x1=2 * i + 1.45, y0=bar, y1=bar,
                          line={"dash": "dot", "color": "#555"}, row=1, col=2)  # fmt: skip
        fig.update_yaxes(title_text="% of the sample", row=1, col=1)
        fig.update_yaxes(title_text="per frame", row=1, col=2)
        pr = clear["pooled_ratio"]
        fig.update_layout(barmode="stack", title=(
            f"R25c: clear non-persons, 448 over 224 {pr[0]:.2f}x [{pr[1]:.2f}, {pr[2]:.2f}] "
            f"({clear['verdict']})"))  # fmt: skip
        figs["r25c_clear"] = fig
    inter = read_json(res / "r7e_interlingua.json")
    if inter:
        pts = [
            {"condition": c, "share of B's own score": v["of_native"],
             "ci": v["of_native_ci"]}
            for c, v in inter["summary"].items()
            if isinstance(v, dict) and "of_native" in v
        ]  # fmt: skip
        figs["interlingua"] = px.bar(
            pd.DataFrame(pts), x="condition", y="share of B's own score", error_y="ci",
            title="Probe: squad A's centre driving squad B's snakes",
        )  # fmt: skip
    kshot = read_json(res / "r13b_kshot.json")
    if kshot:
        pts = [
            {"system": system, "labelled incidents (k)": int(k), "model": name,
             "joint accuracy": m[name][0]}
            for system, r in kshot["systems"].items()
            for k, m in r["by_k"].items()
            for name in ("adapted_joint", "scratch_joint", "frozen_joint", "baro_joint")
        ]  # fmt: skip
        figs["kshot"] = px.line(
            pd.DataFrame(pts), x="labelled incidents (k)", y="joint accuracy",
            color="model", facet_col="system", markers=True,
            title="Probe: the fault specialist on a new system, by labels",
        )  # fmt: skip
    scout = read_json(res / "r15_scout_brain.json")
    if scout:
        pts = [
            {"int8 weights (KB)": r["int8_kb"], "measure": name, "value": r[key],
             "width": r["alpha"]}
            for r in scout
            for key, name in (("ap50", "AP50, all persons"),
                              ("ap50_tall", "AP50, persons ≥ 32 px tall"),
                              ("present_auc", "person-present AUC"))
        ]  # fmt: skip
        figs["r15_scout"] = px.line(
            pd.DataFrame(pts).sort_values("int8 weights (KB)"), x="int8 weights (KB)",
            y="value", color="measure", markers=True, hover_data=["width"],
            title="R15: the scout brain's accuracy by model size (COCO val2017)",
        )  # fmt: skip
    return figs


# ------------------------------------------------------------------ app
def main() -> None:  # pragma: no cover - UI
    import pandas as pd
    import streamlit as st

    st.set_page_config(page_title="genetic-coil-arena", layout="wide")
    st.title("genetic-coil-arena")
    refresh = st.sidebar.selectbox("Live refresh", ["30 s", "60 s", "off"], index=0)
    every = None if refresh == "off" else int(refresh.split()[0])
    st.sidebar.caption(f"Results worktree: `{RESULTS}`")
    st.sidebar.caption(f"Docs: `{DOCS}`")
    res = DOCS / "results"
    figs = build_charts(res)

    story, live, goals_tab, charts, reports = st.tabs(
        ["Research story", "Live runner", "Goal ladder", "All charts", "Reports"]
    )

    # -------------------------------------------------------------- story
    with story:
        log = parse_research_log(
            (DOCS / "research_log.md").read_text(encoding="utf-8", errors="replace")
        )
        st.markdown(log["intro"])
        st.caption(
            "Verdict colours: "
            + " · ".join(
                f":{c}[{label}]"
                for c, label in (
                    ("green", "passed / works"),
                    ("orange", "mixed / not passed"),
                    ("red", "killed"),
                    ("violet", "exploratory probe"),
                    ("blue", "pending"),
                )
            )
        )
        state = read_json(RESULTS / "runner" / "state.json") or {}
        running = [j["id"] for j in state.get("running", [])]
        st.info(
            "Running now: " + (", ".join(running) if running else "nothing")
            + " (details in the Live runner tab)."
        )  # fmt: skip
        colour = {"pass": "green", "mixed": "orange", "kill": "red",
                  "probe": "violet", "pending": "blue"}  # fmt: skip
        for track in log["tracks"]:
            st.header(track["title"])
            st.markdown(track["summary"])
            for e in track["entries"]:
                with st.container(border=True):
                    c = colour.get(e["category"], "gray")
                    st.markdown(
                        f"#### {e['title']}  \n:{c}-background[**{e.get('Verdict', '?')}**]"
                        f" · {e.get('Date', '')}"
                    )
                    left, right = st.columns(2)
                    left.markdown(f"**Question:** {e.get('Question', '')}")
                    left.markdown(f"**Why:** {e.get('Why', '')}")
                    left.markdown(f"**What we did:** {e.get('What we did', '')}")
                    right.markdown(f"**Result:** {e.get('Result', '')}")
                    right.markdown(f"**What it means:** {e.get('What it means', '')}")
                    key = e.get("Chart", "")
                    if key in figs:
                        st.plotly_chart(figs[key], width="stretch", key=f"story-{key}")
                        st.caption("How to read it: " + HOW_TO_READ.get(key, ""))
                    report = e.get("Report", "")
                    if report and (REPO / report).exists():
                        with st.expander(f"Full report ({report})"):
                            st.markdown(
                                (REPO / report).read_text(
                                    encoding="utf-8", errors="replace"
                                )
                            )

    # -------------------------------------------------------------- live
    with live:

        @st.fragment(run_every=every)
        def live_panel() -> None:
            state = read_json(RESULTS / "runner" / "state.json") or {}
            gpu = state.get("gpu") or {}
            c = st.columns(6)
            c[0].metric("Running jobs", len(state.get("running", [])))
            c[1].metric(
                "CPU workers",
                f"{state.get('cpu_workers_in_use', '?')}/{state.get('cpu_budget', '?')}",
            )
            c[2].metric("CPU load", f"{state.get('cpu_percent', 0):.0f}%")
            c[3].metric("GPU util", f"{gpu.get('util', 0):.0f}%")
            c[4].metric(
                "GPU memory",
                f"{gpu.get('mem_used_mb', 0) / 1024:.1f}/{gpu.get('mem_total_mb', 0) / 1024:.0f} GB",
            )
            backlog = state.get("backlog", {})
            c[5].metric(
                "Backlog",
                f"{len(backlog.get('ready', []))} ready, {len(backlog.get('blocked', []))} blocked",
            )
            st.caption(
                f"Runner state updated {state.get('updated_at', '?')}; idle since "
                f"{state.get('idle_since', {})}; draining for update: "
                f"{state.get('draining_for_update')}"
            )
            for job in state.get("running", []):
                st.subheader(f"{job['id']} ({job['lane']}, since {job['started_at']})")
                st.code(
                    tail(RESULTS / "runs" / job["id"] / "log.txt", 12) or "(no log yet)"
                )
            jobs = list_jobs()
            if jobs:
                df = pd.DataFrame(jobs).drop(columns=["dir"])
                st.subheader("Recent jobs")
                st.dataframe(df.head(40), width="stretch", hide_index=True)

        live_panel()
        jobs = list_jobs()
        if jobs:
            pick = st.selectbox("Inspect a job", [j["id"] for j in jobs])
            folder = Path(next(j["dir"] for j in jobs if j["id"] == pick))
            st.code(tail(folder / "log.txt", 60) or tail(folder / "log_tail.txt", 60))
            for md in sorted((folder / "artifacts").glob("*.md")):
                with st.expander(f"Artifact {md.name}"):
                    st.markdown(md.read_text(encoding="utf-8", errors="replace"))

    # -------------------------------------------------------------- goals
    with goals_tab:
        st.caption(
            "Every pre-registered test: its pass bar, its kill bar and its status. "
            "The Research story tab explains them in plain language."
        )
        rows = parse_goals((DOCS / "goals.md").read_text(encoding="utf-8"))
        counts = pd.Series([r["category"] for r in rows]).value_counts()
        cols = st.columns(4)
        for col, (cat, label) in zip(
            cols,
            [("pass", "Passed"), ("mixed", "Mixed / not passed"), ("kill", "Killed"),
             ("pending", "Pending")],
        ):  # fmt: skip
            col.metric(label, int(counts.get(cat, 0)))
        df = pd.DataFrame(rows)[["id", "goal", "verdict", "track", "category"]]

        def paint(row: pd.Series) -> list[str]:
            return [f"color: {CATEGORIES[row['category']]}"] * len(row)

        st.dataframe(df.style.apply(paint, axis=1), width="stretch", hide_index=True)
        for r in rows:
            with st.expander(f"{r['id']}: {r['goal']} ({r['verdict']})"):
                st.markdown(f"**Target:** {r['target']}")
                st.markdown(f"**Kill:** {r['kill']}")
                st.markdown(f"**Status:** {r['status']}")

    # -------------------------------------------------------------- charts
    with charts:
        for key, fig in figs.items():
            st.plotly_chart(fig, width="stretch", key=f"all-{key}")
            st.caption("How to read it: " + HOW_TO_READ.get(key, ""))

    # -------------------------------------------------------------- reports
    with reports:
        files = (
            [DOCS / "research_log.md"]
            + sorted((DOCS / "results").glob("*.md"))
            + sorted((DOCS / "directions").glob("*.md"))
        )
        names = [str(f.relative_to(DOCS)) for f in files]
        pick = st.selectbox("Report", names, index=0 if names else None)
        if pick:
            st.markdown((DOCS / pick).read_text(encoding="utf-8", errors="replace"))


if __name__ == "__main__":
    main()
