"""scripts/queue_job.py: grid expansion, validation and the pushed-commit check."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "queue_job", ROOT / "scripts" / "queue_job.py"
)
qj = importlib.util.module_from_spec(spec)
sys.modules["queue_job"] = qj
spec.loader.exec_module(qj)

ARGS = ["scripts/x.py", "--seed", "{seed}", "--k", "{k}"]


def test_parse_grid():
    assert qj.parse_grid(["seed=1,2", "k=a"]) == {"seed": ["1", "2"], "k": ["a"]}
    for bad in ["seed", "seed=", "=1"]:
        with pytest.raises(ValueError):
            qj.parse_grid([bad])
    with pytest.raises(ValueError):
        qj.parse_grid(["a=1", "a=2"])


def test_grid_expands_cartesian_product():
    jobs = qj.build_jobs(
        "s-{seed}-{k}",
        "c" * 40,
        ARGS,
        qj.parse_grid(["seed=1,2", "k=a,b"]),
        workers=4,
        outputs=["o/{seed}.md"],
        after=["pre-{seed}"],
    )
    assert [j["id"] for j in jobs] == ["s-1-a", "s-1-b", "s-2-a", "s-2-b"]
    j = jobs[1]
    assert j["args"] == ["scripts/x.py", "--seed", "1", "--k", "b"]
    assert j["outputs"] == ["o/1.md"] and j["after"] == ["pre-1"]
    assert j["cpu_workers"] == 4 and j["lane"] == "cpu" and "priority" not in j


def test_single_job_without_grid_keeps_literal_braces():
    (job,) = qj.build_jobs("one", "c" * 40, ["scripts/x.py", "{raw}"], {})
    assert job["args"] == ["scripts/x.py", "{raw}"]
    assert "cpu_workers" not in job  # legacy: runs alone


@pytest.mark.parametrize(
    "kwargs",
    [
        {"job_id": "no-placeholder", "grid": {"a": ["1", "2"]}},
        {"job_id": "bad id {a}", "grid": {"a": ["1"]}},
        {"job_id": "ok", "script": ["other/x.py"]},
        {"job_id": "ok", "script": []},
        {"job_id": "ok", "lane": "tpu"},
        {"job_id": "ok", "workers": -2},
        {"job_id": "dup-{a}", "grid": {"a": ["1", "1"]}},
    ],
)
def test_validation_errors(kwargs):
    with pytest.raises(ValueError):
        qj.build_jobs(
            kwargs["job_id"],
            "c" * 40,
            kwargs.get("script", ["scripts/x.py", "{a}"]),
            kwargs.get("grid", {}),
            lane=kwargs.get("lane", "cpu"),
            workers=kwargs.get("workers"),
        )


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "f").write_text("1")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "one")
    return tmp_path


def test_resolve_commit_requires_pushed(repo):
    sha = _git(repo, "rev-parse", "HEAD").strip()
    with pytest.raises(ValueError, match="not on"):
        qj.resolve_commit(repo, "HEAD", "missing-branch")
    assert qj.resolve_commit(repo, "HEAD", "main") == sha
    (repo / "f").write_text("2")
    _git(repo, "commit", "-qam", "two")
    _git(repo, "branch", "pushed", sha)
    with pytest.raises(ValueError, match="not on"):
        qj.resolve_commit(repo, "HEAD", "pushed")
    with pytest.raises(ValueError, match="unknown"):
        qj.resolve_commit(repo, "nope", "main")


def test_main_writes_files_and_refuses_overwrite(repo, capsys):
    _git(repo, "update-ref", "refs/remotes/origin/jobs", "HEAD")
    argv = ["j-{s}", "--repo", str(repo), "--jobs-branch", "jobs", "--workers", "2"]
    argv += ["--grid", "s=1,2", "--env", "A=b", "--", "scripts/x.py", "{s}"]
    assert qj.main(argv) == 0
    data = json.loads((repo / "runs" / "queue" / "j-2.json").read_text())
    assert data["args"] == ["scripts/x.py", "2"] and data["env"] == {"A": "b"}
    assert qj.main(argv) == 2  # already queued
    assert "already queued" in capsys.readouterr().err
    assert qj.main(["--force", *argv]) == 0
    unpushed = ["x", "--repo", str(repo), "--jobs-branch", "nope", "--", "scripts/a.py"]
    assert qj.main(unpushed) == 2
