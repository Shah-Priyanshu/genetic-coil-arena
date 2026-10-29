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
    ground = read_json(res / "r16_ground_station.json")
    if ground:
        measures = {"exact": "exact command", "asked_when_unclear": "asked when unclear",
                    "refused_when_should": "refused when should",
                    "wrong_commands": "wrong command"}  # fmt: skip
        pts = [
            {"translator": r["translator"].replace("ollama:", ""), "measure": label,
             "%": 100 * r[key][0]}
            for name, r in ground["runs"].items() if name.startswith("sealed|")
            for key, label in measures.items()
        ]  # fmt: skip
        fig = px.bar(
            pd.DataFrame(pts), x="translator", y="%", color="measure", barmode="group",
            color_discrete_sequence=["#2e7d32", "#66bb6a", "#a5d6a7", "#c62828"],
            title="R16: the ground station on the sealed requests",
        )  # fmt: skip
        fig.add_hline(y=3, line_dash="dash", line_color="#c62828")
        figs["r16_ground"] = fig
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
