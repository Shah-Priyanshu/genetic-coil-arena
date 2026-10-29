"""Dashboard helpers (stdlib only; the Streamlit app itself is not run here)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "dashboard", ROOT / "scripts" / "dashboard.py"
)
dashboard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dashboard)


def test_parse_goals_reads_the_real_ladder():
    rows = dashboard.parse_goals(
        (ROOT / "docs" / "goals.md").read_text(encoding="utf-8")
    )
    ids = [r["id"] for r in rows]
    assert "G0" in ids and "G7c" in ids and len(ids) == len(set(ids))
    by_id = {r["id"]: r for r in rows}
    assert by_id["G7c"]["category"] == "pass"
    assert by_id["G4"]["category"] == "kill"


def test_verdicts_and_categories():
    assert dashboard.verdict_of("**passed** (2026-10-29). More.") == "passed"
    assert (
        dashboard.verdict_of("pre-registered (2026-10-29); job x") == "pre-registered"
    )
    assert dashboard.category_of("inconclusive, leaning kill") == "mixed"
    assert dashboard.category_of("kill by the pre-registered rule") == "kill"
    assert dashboard.category_of("radio does not bind") == "mixed"


def test_list_jobs_and_tail(tmp_path):
    job = tmp_path / "runs" / "a"
    job.mkdir(parents=True)
    (job / "status.json").write_text('{"id": "a", "state": "done", "updated_at": "1"}')
    (job / "log.txt").write_text("x\ny\nz\n")
    jobs = dashboard.list_jobs(tmp_path)
    assert jobs[0]["id"] == "a" and jobs[0]["state"] == "done"
    assert dashboard.tail(job / "log.txt", 2) == "y\nz"


def test_research_log_parses_and_cites_existing_reports():
    log = dashboard.parse_research_log(
        (ROOT / "docs" / "research_log.md").read_text(encoding="utf-8")
    )
    assert "end goal" in log["intro"].lower()
    entries = [e for t in log["tracks"] for e in t["entries"]]
    assert len(log["tracks"]) >= 5 and len(entries) >= 15
    for e in entries:
        for field in (
            "Question",
            "Why",
            "What we did",
            "Result",
            "Verdict",
            "What it means",
        ):
            assert e.get(field), (e["title"], field)
        assert (ROOT / e["Report"]).exists(), e["Report"]
    assert {e["category"] for e in entries} >= {"pass", "kill", "probe"}
