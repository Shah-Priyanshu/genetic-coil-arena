"""Run queued experiment jobs on a GPU machine and report back through git.

The queue lives in ``runs/queue/*.json`` on a jobs branch. Each job is pinned to a
commit and runs in a fresh detached worktree of that commit, so the runner never
touches your working checkout. Progress (status, log tail, GPU stats) and final
outputs are pushed to an orphan results branch, ``runs/<job id>/``.

    python scripts/gpu_runner.py --python D:/venvs/gca/Scripts/python.exe

Job file (``runs/queue/<id>.json``)::

    {
      "id": "r6-gpu-001",
      "commit": "<full sha on the jobs branch>",
      "args": ["scripts/learned_protocol.py", "--scale", "gpu"],
      "timeout_min": 240,
      "env": {"EXTRA": "1"},
      "outputs": ["docs/results/learned_protocol.md"]
    }

``args[0]`` must be a script under ``scripts/``. The job also gets ``GCA_RUN_DIR``,
a directory whose contents are pushed with the results. The runner only executes
commits reachable from the jobs branch, one job at a time, and skips any job that
already has a final status on the results branch. To stop a running job, push an
empty file ``runs/cancel/<job id>`` to the jobs branch; it is noticed within
``--push-every`` seconds. Stop the runner itself with Ctrl+C.
"""

import argparse
import json
import os
import platform
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

FINAL_STATES = ("done", "failed", "timeout", "cancelled", "rejected")
LOG_TAIL_LINES = 200
MAX_LOG_BYTES = 5_000_000


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def gpu_stats() -> Optional[str]:
    """One line per GPU from nvidia-smi, or None if it isn't available."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


class Results:
    """A worktree on the orphan results branch, written to and pushed by the runner."""

    def __init__(self, repo: Path, root: Path, branch: str, remote: str) -> None:
        self.repo, self.root, self.branch, self.remote = repo, root, branch, remote
        if not (root / ".git").exists():
            self._create()
        self.pull()

    def _create(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        git(self.repo, "worktree", "prune")
        has_remote = git(
            self.repo, "ls-remote", "--heads", self.remote, self.branch
        ).strip()
        if has_remote:
            git(self.repo, "fetch", self.remote, self.branch)
            git(
                self.repo,
                "worktree",
                "add",
                "-B",
                self.branch,
                str(self.root),
                f"{self.remote}/{self.branch}",
            )
        else:
            git(self.repo, "worktree", "add", "--detach", str(self.root))
            git(self.root, "checkout", "--orphan", self.branch)
            git(self.root, "rm", "-rf", "--quiet", ".", check=False)
            (self.root / "README.md").write_text(
                "Results pushed by scripts/gpu_runner.py. One folder per job under runs/.\n",
                encoding="utf-8",
            )
            self.commit_and_push("Start results branch")

    def pull(self) -> None:
        if git(self.repo, "ls-remote", "--heads", self.remote, self.branch).strip():
            git(self.root, "pull", "--rebase", "--quiet", self.remote, self.branch)

    def job_dir(self, job_id: str) -> Path:
        path = self.root / "runs" / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def final_state(self, job_id: str) -> Optional[str]:
        status = self.root / "runs" / job_id / "status.json"
        if not status.exists():
            return None
        state = json.loads(status.read_text(encoding="utf-8")).get("state")
        return state if state in FINAL_STATES else None

    def commit_and_push(self, message: str) -> None:
        git(self.root, "add", "-A")
        if git(self.root, "status", "--porcelain").strip():
            git(self.root, "commit", "-q", "-m", message)
        for attempt in range(4):
            try:
                git(self.root, "push", "-q", "-u", self.remote, self.branch)
                return
            except RuntimeError as err:
                print(f"push failed ({err}); retrying", file=sys.stderr)
                time.sleep(2 ** (attempt + 1))
                self.pull()
        print("giving up on this push; will retry at the next update", file=sys.stderr)


def cancel_requested(repo: Path, args: argparse.Namespace, job_id: str) -> bool:
    """True if ``runs/cancel/<job id>`` exists on the jobs branch."""
    git(repo, "fetch", "-q", args.remote, args.jobs_branch, check=False)
    found = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "cat-file",
            "-e",
            f"{args.jobs_ref}:runs/cancel/{job_id}",
        ],
        capture_output=True,
    )
    return found.returncode == 0


def queued_jobs(repo: Path, ref: str) -> list[dict[str, Any]]:
    names = git(repo, "ls-tree", "--name-only", f"{ref}:runs/queue", check=False)
    jobs = []
    for name in sorted(names.split()):
        if not name.endswith(".json"):
            continue
        try:
            job = json.loads(git(repo, "show", f"{ref}:runs/queue/{name}"))
        except (RuntimeError, json.JSONDecodeError) as err:
            print(f"skipping {name}: {err}", file=sys.stderr)
            continue
        job.setdefault("id", name[: -len(".json")])
        jobs.append(job)
    return jobs


def validate(repo: Path, ref: str, job: dict[str, Any]) -> Optional[str]:
    """Why the job can't run, or None if it can."""
    args = job.get("args")
    if (
        not isinstance(args, list)
        or not args
        or not all(isinstance(a, str) for a in args)
    ):
        return "args must be a non-empty list of strings"
    script = args[0].replace("\\", "/")
    if not script.startswith("scripts/") or ".." in script.split("/"):
        return "args[0] must be a script under scripts/"
    commit = job.get("commit")
    if not isinstance(commit, str) or not commit:
        return "commit missing"
    check = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", commit, ref],
        capture_output=True,
    )
    if check.returncode != 0:
        return f"commit {commit} is not on {ref}"
    return None


def write_status(path: Path, **fields: Any) -> None:
    path.write_text(json.dumps(fields, indent=2) + "\n", encoding="utf-8")


def run_job(args: argparse.Namespace, repo: Path, results: Results, job: dict) -> None:
    job_id = job["id"]
    out_dir = results.job_dir(job_id)
    (out_dir / "job.json").write_text(
        json.dumps(job, indent=2) + "\n", encoding="utf-8"
    )
    status = {"id": job_id, "commit": job.get("commit"), "queued_args": job.get("args")}

    reason = validate(repo, args.jobs_ref, job)
    if reason:
        write_status(out_dir / "status.json", **status, state="rejected", reason=reason)
        results.commit_and_push(f"{job_id}: rejected")
        print(f"{job_id}: rejected ({reason})")
        return

    work = Path(tempfile.mkdtemp(prefix=f"gca-{job_id}-"))
    git(repo, "worktree", "add", "--detach", str(work), job["commit"])
    run_dir = out_dir / "artifacts"
    run_dir.mkdir(exist_ok=True)
    env = {
        **os.environ,
        **{k: str(v) for k, v in job.get("env", {}).items()},
        "PYTHONPATH": str(work / "src"),
        "PYTHONUTF8": "1",
        "PYTHONUNBUFFERED": "1",
        "GCA_RUN_DIR": str(run_dir),
    }
    env.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "gca_numba"))
    log_path = out_dir / "log.txt"
    tail: deque = deque(maxlen=LOG_TAIL_LINES)
    started = time.monotonic()
    timeout = float(job.get("timeout_min", args.default_timeout_min)) * 60
    print(f"{job_id}: running {' '.join(job['args'])} at {job['commit'][:10]}")

    def push_progress(state: str, **extra: Any) -> None:
        (out_dir / "log_tail.txt").write_text("".join(tail), encoding="utf-8")
        write_status(
            out_dir / "status.json",
            **status,
            state=state,
            started_at=started_at,
            updated_at=now(),
            elapsed_min=round((time.monotonic() - started) / 60, 1),
            gpu=gpu_stats(),
            host=platform.node(),
            **extra,
        )
        results.commit_and_push(f"{job_id}: {state}")

    started_at = now()
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(
            [args.python, *job["args"]],
            cwd=work,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        lines: queue.Queue[Optional[str]] = queue.Queue()

        def read_output() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                lines.put(line)
            lines.put(None)

        threading.Thread(target=read_output, daemon=True).start()
        push_progress("running")
        last_push = time.monotonic()
        state = "done"
        while True:
            try:
                line = lines.get(timeout=5)
            except queue.Empty:
                line = ""
            if line is None:
                break
            if line:
                if log.tell() < MAX_LOG_BYTES:
                    log.write(line)
                    log.flush()
                tail.append(line)
            if time.monotonic() - started > timeout:
                proc.kill()
                state = "timeout"
                break
            if time.monotonic() - last_push > args.push_every:
                if cancel_requested(repo, args, job_id):
                    proc.kill()
                    state = "cancelled"
                    break
                push_progress("running")
                last_push = time.monotonic()
        returncode = proc.wait()
    if state == "done" and returncode != 0:
        state = "failed"

    for rel in job.get("outputs", []):
        src = work / rel
        dest = out_dir / "outputs" / rel
        if src.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        elif src.is_dir():
            shutil.copytree(src, dest, dirs_exist_ok=True)
    push_progress(state, returncode=returncode, finished_at=now())
    git(repo, "worktree", "remove", "--force", str(work), check=False)
    print(f"{job_id}: {state} (exit {returncode})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Interpreter for jobs (a venv with the project's dependencies and torch)",
    )
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--jobs-branch", default="claude/vibrant-hawking-zodb5a")
    parser.add_argument("--results-branch", default="gpu-results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Where to keep the results worktree (default: next to the repo)",
    )
    parser.add_argument(
        "--poll", type=int, default=60, help="Seconds between queue checks"
    )
    parser.add_argument(
        "--push-every", type=int, default=300, help="Seconds between progress pushes"
    )
    parser.add_argument("--default-timeout-min", type=float, default=240)
    parser.add_argument("--once", action="store_true", help="Run pending jobs and exit")
    args = parser.parse_args()

    repo = args.repo.resolve()
    args.jobs_ref = f"{args.remote}/{args.jobs_branch}"
    results_dir = args.results_dir or repo.parent / f"{repo.name}-gpu-results"
    git(repo, "fetch", "-q", args.remote, args.jobs_branch)
    results = Results(repo, results_dir.resolve(), args.results_branch, args.remote)
    print(f"watching {args.jobs_ref}:runs/queue; results -> {args.results_branch}")
    print(f"GPU: {gpu_stats() or 'nvidia-smi not found'}")

    while True:
        git(repo, "fetch", "-q", args.remote, args.jobs_branch)
        results.pull()
        for job in queued_jobs(repo, args.jobs_ref):
            if results.final_state(job["id"]) is not None:
                continue
            if cancel_requested(repo, args, job["id"]):
                print(f"{job['id']}: cancelled before start; skipping")
                continue
            run_job(args, repo, results, job)
        if args.once:
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
