"""Exit conditions and job tracking of scripts/runner_watch.py."""

import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "runner_watch", ROOT / "scripts" / "runner_watch.py"
)
watch_mod = importlib.util.module_from_spec(spec)
sys.modules["runner_watch"] = watch_mod
spec.loader.exec_module(watch_mod)


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat()


def state(now, cpu_idle_min=None, gpu_idle_min=None, updated_ago_min=0.5, running=()):
    def since(m):
        return None if m is None else iso(now - 60 * m)

    return {
        "updated_at": iso(now - 60 * updated_ago_min),
        "idle_since": {"cpu": since(cpu_idle_min), "gpu": since(gpu_idle_min)},
        "running": [{"id": j, "lane": lane} for j, lane in running],
        "backlog": {"ready": [], "blocked": []},
    }


class TestCheck:
    now = 1_800_000_000.0
    busy = {"cpu": True, "gpu": True}
    never = {"cpu": False, "gpu": False}

    def check(self, st, was_busy):
        return watch_mod.check(st, self.now, was_busy, 10, 20)

    def test_busy_pc_is_fine(self):
        assert self.check(state(self.now), self.busy) is None

    def test_both_lanes_idle_is_pc_idle_even_if_never_busy(self):
        got = self.check(state(self.now, 12, 30), self.never)
        assert got[0] == "pc-idle"

    def test_lane_idle_only_after_it_was_busy(self):
        st = state(self.now, cpu_idle_min=None, gpu_idle_min=15)
        assert self.check(st, self.never) is None
        assert self.check(st, self.busy)[0] == "lane-idle"

    def test_short_idle_waits(self):
        assert self.check(state(self.now, 5, 5), self.busy) is None

    def test_stale_state(self):
        assert self.check(state(self.now, updated_ago_min=25), self.busy)[0] == "stale"
        assert self.check({}, self.busy)[0] == "stale"


class TestCiProblem:
    def summary(self, state="done", counts="1255 passed, 1 skipped"):
        return f"commit abc\nstate {state}\ncounts: {counts}\n\nlog..."

    def test_clean_pass(self):
        assert watch_mod.ci_problem(self.summary()) is None

    def test_failures_and_errors(self):
        assert watch_mod.ci_problem(self.summary(counts="2 failed, 10 passed"))
        assert watch_mod.ci_problem(self.summary(counts="10 passed, 1 errors"))

    def test_missing_counts_or_bad_state(self):
        assert watch_mod.ci_problem(self.summary(counts="no counts found"))
        assert watch_mod.ci_problem(self.summary(state="timeout")) == "state timeout"


class TestWatch:
    def make(self, tmp_path, now):
        (tmp_path / "runner").mkdir()
        st = state(now, running=[("a", "cpu")])
        (tmp_path / "runner" / "state.json").write_text(json.dumps(st))
        return watch_mod.Watch(tmp_path)

    def finish(self, root, job_id, state_name="done", commit="c1"):
        folder = root / "runs" / job_id
        folder.mkdir(parents=True)
        status = {
            "id": job_id,
            "state": state_name,
            "elapsed_min": 1.0,
            "commit": commit,
            "finished_at": iso(time.time()),
        }
        (folder / "status.json").write_text(json.dumps(status))

    def test_reports_end_once_and_flags_failures(self, tmp_path, capsys):
        w = self.make(tmp_path, time.time())
        assert w.poll() is None
        assert w.was_busy["cpu"] and not w.was_busy["gpu"]
        self.finish(tmp_path, "ok-job")
        assert w.poll() is None
        self.finish(tmp_path, "bad-job", "failed")
        reason, details = w.poll()
        assert reason == "failure" and "bad-job" in details
        out = capsys.readouterr().out
        assert out.count("end ok-job done") == 1 and "start a (cpu)" in out

    def test_ci_failure_is_flagged(self, tmp_path):
        w = self.make(tmp_path, time.time())
        folder = tmp_path / "ci" / "c1"
        folder.mkdir(parents=True)
        (folder / "summary.txt").write_text(
            "commit c1\nstate done\ncounts: 1 failed, 9 passed\n"
        )
        self.finish(tmp_path, "ci-c1")
        reason, details = w.poll()
        assert reason == "failure" and "1 failed" in details

    def test_old_results_are_ignored(self, tmp_path):
        w = self.make(tmp_path, time.time())
        w.started = time.time() + 3600
        self.finish(tmp_path, "old-job", "failed")
        assert w.poll() is None
