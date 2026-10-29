"""Scheduling, dependencies, CI debounce and bookkeeping of scripts/gpu_runner.py."""

import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load("gpu_runner")


def J(id, workers=None, lane="cpu", priority=0, **extra):
    job = {"id": id, "lane": lane, "priority": priority, **extra}
    if workers is not None:
        job["cpu_workers"] = workers
    return job


def ids(jobs):
    return [j["id"] for j in jobs]


class TestPickJobs:
    def test_fills_budget_in_priority_then_id_order(self):
        queued = [J("b", 8), J("a", 8), J("c", 8, priority=5), J("d", 8)]
        assert ids(runner.pick_jobs(queued, [], 24)) == ["c", "a", "b"]

    def test_running_workers_count_against_budget(self):
        running = [J("r", 20)]
        assert ids(runner.pick_jobs([J("a", 4), J("b", 5)], running, 24)) == ["a"]

    def test_small_job_backfills_past_one_that_does_not_fit(self):
        queued = [J("big", 20, priority=1), J("small", 2)]
        assert ids(runner.pick_jobs(queued, [J("r", 10)], 24)) == ["small"]

    def test_one_gpu_job_at_a_time(self):
        queued = [J("g1", 2, "gpu"), J("g2", 2, "gpu"), J("c", 2)]
        assert ids(runner.pick_jobs(queued, [], 24)) == ["c", "g1"]
        assert ids(runner.pick_jobs(queued, [J("g0", 2, "gpu")], 24)) == ["c"]

    def test_gpu_reserve_keeps_gpu_lane_free_when_cpu_lane_is_full(self):
        running = [J("c1", 8), J("c2", 8), J("c3", 8)]
        queued = [J("g", 4, "gpu"), J("c4", 8)]
        assert ids(runner.pick_jobs(queued, running, 24)) == []
        assert ids(runner.pick_jobs(queued, running, 24, gpu_reserve=4)) == ["g"]

    def test_gpu_workers_beyond_reserve_count_against_budget(self):
        queued = [J("c", 20, priority=1), J("g", 8, "gpu")]
        assert ids(runner.pick_jobs(queued, [], 24, gpu_reserve=4)) == ["c", "g"]
        queued = [J("c", 21, priority=1), J("g", 8, "gpu")]
        assert ids(runner.pick_jobs(queued, [], 24, gpu_reserve=4)) == ["c"]
        running = [J("g", 8, "gpu")]
        assert ids(runner.pick_jobs([J("c", 21)], running, 24, gpu_reserve=4)) == []

    def test_gpu_and_cpu_overlap(self):
        assert ids(runner.pick_jobs([J("c", 20)], [J("g", 0, "gpu")], 24)) == ["c"]

    def test_legacy_job_runs_alone_and_blocks_others(self):
        assert runner.pick_jobs([J("a", 2)], [J("old")], 24) == []
        assert ids(runner.pick_jobs([J("old")], [], 24)) == ["old"]
        assert runner.pick_jobs([J("old")], [J("r", 2)], 24) == []

    def test_waiting_legacy_job_stops_lower_priority_starts(self):
        queued = [J("old", priority=3), J("a", 2)]
        assert runner.pick_jobs(queued, [J("r", 2)], 24) == []
        # but higher-priority jobs ahead of it still start
        queued = [J("old", priority=0), J("hi", 2, priority=9)]
        assert ids(runner.pick_jobs(queued, [], 24)) == ["hi"]

    def test_legacy_job_first_when_machine_free(self):
        queued = [J("old", priority=1), J("a", 2)]
        assert ids(runner.pick_jobs(queued, [], 24)) == ["old"]

    def test_oversized_job_is_clamped_to_budget(self):
        assert ids(runner.pick_jobs([J("a", 99)], [], 24)) == ["a"]

    def test_zero_worker_gpu_job(self):
        assert ids(runner.pick_jobs([J("g", 0, "gpu")], [J("r", 24)], 24)) == ["g"]

    def test_bad_fields_fall_back_to_defaults(self):
        job = {"id": "x", "cpu_workers": True, "priority": "high", "lane": "tpu"}
        assert runner.is_exclusive(job)
        assert runner.job_lane(job) == "cpu" and runner.job_priority(job) == 0


class TestClassify:
    def test_ready_blocked_rejected(self):
        jobs = [
            J("a"),
            J("b", after=["a"]),
            J("c", after=["x"]),
            J("d", after=["ok"]),
            J("e", after=["ok", "bad"]),
        ]
        ready, blocked, rejected = runner.classify(
            jobs, {"ok": "done", "bad": "failed"}
        )
        assert ids(ready) == ["a", "d"]
        assert blocked == {"b": ["a"], "c": ["x"]}
        assert rejected == {"e": "dependency bad ended failed"}

    @pytest.mark.parametrize("state", ["failed", "timeout", "cancelled", "rejected"])
    def test_any_non_done_final_state_rejects(self, state):
        _, _, rejected = runner.classify([J("b", after=["a"])], {"a": state})
        assert "b" in rejected

    def test_dependency_in_flight_blocks(self):
        _, blocked, rejected = runner.classify([J("b", after=["a"])], {})
        assert blocked == {"b": ["a"]} and not rejected


class TestCi:
    interval = 1800

    def test_new_head_queues_when_never_run(self):
        assert runner.ci_next("h1", None, None, None, 1000, self.interval) == "h1"

    def test_same_head_as_last_run_does_nothing(self):
        assert runner.ci_next("h1", "h1", 0, None, 10**6, self.interval) is None

    def test_debounce_within_interval(self):
        assert runner.ci_next("h2", "h1", 1000, None, 1100, self.interval) is None

    def test_queues_after_interval(self):
        assert runner.ci_next("h2", "h1", 1000, None, 2900, self.interval) == "h2"

    def test_pending_job_retargets_to_latest_head(self):
        assert runner.ci_next("h3", "h1", 1000, "h2", 1100, self.interval) == "h3"

    def test_pending_kept_when_head_unreadable(self):
        assert runner.ci_next(None, "h1", 0, "h2", 5000, self.interval) == "h2"

    def test_ci_job_shape(self):
        job = runner.ci_job("a" * 40)
        assert job["id"] == "ci-" + "a" * 12
        assert job["priority"] < -100 and job["cpu_workers"] == 2
        assert job["timeout_min"] == 60 and job["lane"] == "cpu"
        assert "tests/test_viewer.py" in " ".join(job["args"])

    def test_ci_job_is_lowest_priority(self):
        queued = [runner.ci_job("a" * 40), J("r", 4, priority=-5)]
        assert ids(runner.pick_jobs(queued, [], 24)) == ["r", "ci-" + "a" * 12]

    def test_pytest_counts(self):
        text = "...\n== 2 failed, 120 passed, 3 skipped in 45.20s ==\n"
        assert runner.parse_pytest_counts(text) == {
            "failed": 2,
            "passed": 120,
            "skipped": 3,
        }
        assert runner.parse_pytest_counts("123 passed in 1.0s") == {"passed": 123}
        assert runner.parse_pytest_counts("nothing here") == {}


class TestAggregator:
    def test_emits_row_when_hour_changes(self):
        agg = runner.HourlyAggregator()

        def t(h, m):
            return datetime(2026, 1, 1, h, m, tzinfo=timezone.utc)

        gpu = {"util": 60.0, "mem_used_mb": 1000.0, "mem_total_mb": 12000.0}
        assert agg.add(t(1, 0), 10.0, gpu, True, False, 4) is None
        assert agg.add(t(1, 30), 30.0, None, False, False, 0) is None
        row = agg.add(t(2, 0), 0.0, None, False, False, 0)
        assert row["hour_utc"] == "2026-01-01T01:00Z" and row["samples"] == 2
        assert row["cpu_mean_pct"] == "20.0" and row["gpu_util_mean_pct"] == "60.0"
        assert row["cpu_lane_busy_frac"] == "0.500"

    def test_missing_cpu_is_blank(self):
        agg = runner.HourlyAggregator()
        when = datetime(2026, 1, 1, 1, tzinfo=timezone.utc)
        agg.add(when, None, None, False, False, 0)
        assert agg.flush()["cpu_mean_pct"] == ""


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "jobs")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "a.txt").write_text("1")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "one")
    return tmp_path


class TestValidateAndQueue:
    def job(self, repo, **extra):
        sha = _git(repo, "rev-parse", "HEAD").strip()
        return {"id": "x", "commit": sha, "args": ["scripts/a.py"], **extra}

    def test_accepts_old_style_job(self, repo):
        assert runner.validate(repo, "HEAD", self.job(repo)) is None

    def test_accepts_new_fields(self, repo):
        job = self.job(repo, lane="gpu", cpu_workers=3, priority=-2, after=["a"])
        assert runner.validate(repo, "HEAD", job) is None

    @pytest.mark.parametrize(
        "extra",
        [
            {"lane": "tpu"},
            {"cpu_workers": "4"},
            {"cpu_workers": -1},
            {"priority": 1.5},
            {"after": "a"},
            {"args": ["other/a.py"]},
        ],
    )
    def test_rejects_bad_fields(self, repo, extra):
        assert runner.validate(repo, "HEAD", self.job(repo, **extra))

    def test_queue_file_cannot_claim_internal(self, repo):
        (repo / "runs" / "queue").mkdir(parents=True)
        job = {"id": "x", "internal": "ci", "args": ["-c", "print(1)"], "commit": "z"}
        (repo / "runs" / "queue" / "x.json").write_text(json.dumps(job))
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "q")
        (loaded,) = runner.queued_jobs(repo, "HEAD")
        assert "internal" not in loaded
        assert runner.validate(repo, "HEAD", loaded)

    def test_cancelled_ids(self, repo):
        (repo / "runs" / "cancel").mkdir(parents=True)
        (repo / "runs" / "cancel" / "k").write_text("")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "c")
        assert runner.cancelled_ids(repo, "HEAD") == {"k"}

    def test_runner_version_is_blob_sha(self, repo):
        (repo / "r.py").write_text("print(1)\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "r")
        expected = _git(repo, "rev-parse", "HEAD:r.py").strip()
        assert runner.runner_version(repo / "r.py") == expected


@pytest.mark.parametrize("use_psutil", [True, False])
def test_kill_tree_kills_children(use_psutil, monkeypatch):
    if use_psutil and runner.psutil is None:
        pytest.skip("psutil not installed")
    if not use_psutil:
        if sys.platform != "win32":
            pytest.skip("the fallback is taskkill")
        monkeypatch.setattr(runner, "psutil", None)
    code = (
        "import subprocess, sys, time\n"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "print(c.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code], stdout=subprocess.PIPE, text=True
    )
    child = int(proc.stdout.readline())
    runner.kill_tree(proc)
    proc.wait(timeout=10)
    proc.stdout.close()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and _alive(child):
        time.sleep(0.2)
    assert not _alive(child)


def _alive(pid: int) -> bool:
    if runner.psutil is not None:
        return runner.psutil.pid_exists(pid)
    out = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True
    ).stdout
    return str(pid) in out
