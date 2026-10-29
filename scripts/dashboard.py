"""Live project dashboard (Streamlit): runner status, goal ladder, result charts and reports.

    powershell -File scripts/start_dashboard.ps1        # own window, http://localhost:8501
    D:/venvs/dash/Scripts/streamlit run scripts/dashboard.py

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
}


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
    if v.startswith(("pre-registered", "planned", "in progress", "running")):
        return "pending"
    if "leaning kill" in v or "inconclusive" in v or "not passed" in v:
        return "mixed"
    if v.startswith("kill"):
        return "kill"
    if v.startswith(("passed", "pass", "met")):
        return "pass"
    return "mixed"


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


# ------------------------------------------------------------------ app
def main() -> None:  # pragma: no cover - UI
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st

    st.set_page_config(page_title="genetic-coil-arena", layout="wide")
    st.title("genetic-coil-arena")
    refresh = st.sidebar.selectbox("Live refresh", ["30 s", "60 s", "off"], index=0)
    every = None if refresh == "off" else int(refresh.split()[0])
    st.sidebar.caption(f"Results worktree: `{RESULTS}`")
    st.sidebar.caption(f"Docs: `{DOCS}`")

    live, goals_tab, charts, reports = st.tabs(
        ["Live runner", "Goal ladder", "Results", "Reports"]
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

        def colour(row: pd.Series) -> list[str]:
            return [f"color: {CATEGORIES[row['category']]}"] * len(row)

        st.dataframe(df.style.apply(colour, axis=1), width="stretch", hide_index=True)
        for r in rows:
            with st.expander(f"{r['id']}: {r['goal']} ({r['verdict']})"):
                st.markdown(f"**Target:** {r['target']}")
                st.markdown(f"**Kill:** {r['kill']}")
                st.markdown(f"**Status:** {r['status']}")

    # -------------------------------------------------------------- charts
    with charts:
        res = DOCS / "results"
        st.header("A learned centre follows goals (G7, G7c)")
        curves = read_csv(res / "r7b_goal_conditioned_16runs_curves.csv")
        if curves:
            cdf = pd.DataFrame(curves)
            cdf = cdf[cdf["val_mission"] != ""].astype(
                {"generation": int, "val_mission": float}
            )
            agg = (
                cdf.groupby(["mode", "generation"])["val_mission"]
                .quantile([0.25, 0.5, 0.75])
                .unstack()
                .reset_index()
            )
            fig = go.Figure()
            for mode, part in agg.groupby("mode"):
                fig.add_trace(go.Scatter(x=part["generation"], y=part[0.5], name=mode))
            fig.update_layout(
                title="Validation mission score while training (median of 16 runs)",
                xaxis_title="generation",
                yaxis_title="mission score",
            )
            st.plotly_chart(fig, width="stretch")
        ref = read_json(res / "r7b_goal_conditioned_16runs.json")
        r7c = read_json(res / "r7c_reliability.json")
        if ref:
            bars = [
                {"condition": c["name"], "held-out score": c["mean"], "ci": c["ci"]}
                for c in ref["sets"]["heldout"]["result"]["conditions"]
            ]
            if r7c:
                kept = [r["heldout"] for r in r7c["per_run"] if r["kept"]]
                n = len(kept)
                m = sum(kept) / n
                sd = (sum((x - m) ** 2 for x in kept) / (n - 1)) ** 0.5
                bars.append({"condition": "(b) fresh runs, G7c", "held-out score": m,
                             "ci": 2.13 * sd / n**0.5})  # fmt: skip
            bdf = pd.DataFrame(bars)
            st.plotly_chart(
                px.bar(bdf, x="condition", y="held-out score", error_y="ci",
                       title="Held-out goals (mean ± 95% CI over runs)"),
                width="stretch",
            )  # fmt: skip
        st.header("Reading and writing the learned language (G12)")
        r7d = read_json(res / "r7d_legibility.json")
        if r7d:
            pts = [
                {"condition": r["mode"], "run": r["run"],
                 "eater set read": r["read"]["eaters_exact"],
                 "write efficacy": r["write"]["efficacy"]}
                for r in r7d["runs"]
                if r["write"]["efficacy"] == r["write"]["efficacy"]
            ]  # fmt: skip
            st.plotly_chart(
                px.scatter(pd.DataFrame(pts), x="eater set read", y="write efficacy",
                           color="condition", hover_data=["run"],
                           title="Per squad: read accuracy vs write efficacy (held-out goals)"),
                width="stretch",
            )  # fmt: skip
        st.header("Learned codes where bandwidth binds (G2, plant floor)")
        plant = read_json(res / "plant_learned_codes.json")
        if plant:
            pdf = pd.DataFrame([r for r in plant["rows"] if r.get("variant") == "base"])
            line = pdf.groupby(["encoding", "budget"])["macro_f1"].mean().reset_index()
            hand = {
                t["budget"]: t["hand_mean"] for t in plant["g2_tests"]
            }  # best hand design per budget
            line = pd.concat(
                [line, pd.DataFrame([{"encoding": "best hand design", "budget": b,
                                      "macro_f1": v} for b, v in hand.items()])]
            )  # fmt: skip
            st.plotly_chart(
                px.line(line, x="budget", y="macro_f1", color="encoding", markers=True,
                        log_x=True, title="Fault diagnosis macro-F1 by bytes per uplink"),
                width="stretch",
            )  # fmt: skip
        st.header("Drone swarm step 1 (G13)")
        swarm = read_json(res / "r14_swarm_talk.json")
        if swarm:
            pts = []
            for c in swarm["cells"].values():
                s = c["success"]
                pts.append({"budget": c["budget"], "loss": c["loss"], "drift": c["drift"],
                            "success": sum(s) / len(s)})  # fmt: skip
            sdf = pd.DataFrame(pts)
            silent = sdf[sdf["budget"] == 0]
            talk = sdf[sdf["budget"] > 0].copy()
            talk["setting"] = (
                "loss "
                + talk["loss"].astype(str)
                + ", drift "
                + talk["drift"].astype(str)
            )
            fig = px.line(talk, x="budget", y="success", color="setting", markers=True,
                          log_x=True, title="Source found within 2 min, by bytes per packet")  # fmt: skip
            for _, r in silent.iterrows():
                fig.add_hline(y=r["success"], line_dash="dot",
                              annotation_text=f"silent, drift {r['drift']}")  # fmt: skip
            st.plotly_chart(fig, width="stretch")

        st.header("Probes (exploratory, 2026-10-29)")
        multi = read_json(res / "r14_multi_probe.json")
        if multi:
            pts = [
                {"sources": key.split("|")[0], "system": key.split("|")[1],
                 "found": v["found"][0], "ci": v["found"][1], "all": v["all"][0]}
                for key, v in multi["summary"].items()
                if key.count("|") == 1
            ]  # fmt: skip
            st.plotly_chart(
                px.bar(pd.DataFrame(pts), x="system", y="found", error_y="ci",
                       color="sources", barmode="group", hover_data=["all"],
                       title="Swarm with several gas sources: sources found in 2 min"),
                width="stretch",
            )  # fmt: skip
        inter = read_json(res / "r7e_interlingua.json")
        if inter:
            pts = [
                {
                    "condition": c,
                    "share of native": v["of_native"],
                    "ci": v["of_native_ci"],
                }
                for c, v in inter["summary"].items()
                if isinstance(v, dict) and "of_native" in v
            ]
            st.plotly_chart(
                px.bar(pd.DataFrame(pts), x="condition", y="share of native", error_y="ci",
                       title="Squad A's centre driving squad B through the meaning space"),
                width="stretch",
            )  # fmt: skip
        kshot = read_json(res / "r13b_kshot.json")
        if kshot:
            pts = []
            for system, r in kshot["systems"].items():
                for k, m in r["by_k"].items():
                    for name in ("adapted_joint", "scratch_joint", "frozen_joint",
                                 "baro_joint"):  # fmt: skip
                        pts.append({"system": system, "k": int(k), "model": name,
                                    "joint": m[name][0]})  # fmt: skip
            st.plotly_chart(
                px.line(pd.DataFrame(pts), x="k", y="joint", color="model",
                        facet_col="system", markers=True,
                        title="Fault specialist on a new system: joint accuracy vs labelled incidents"),
                width="stretch",
            )  # fmt: skip

    # -------------------------------------------------------------- reports
    with reports:
        files = sorted((DOCS / "results").glob("*.md")) + sorted(
            (DOCS / "directions").glob("*.md")
        )
        names = [str(f.relative_to(DOCS)) for f in files]
        pick = st.selectbox("Report", names, index=0 if names else None)
        if pick:
            st.markdown((DOCS / pick).read_text(encoding="utf-8", errors="replace"))


if __name__ == "__main__":
    main()
