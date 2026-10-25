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
      "outputs": ["docs/results/learned_protocol.md"],
      "lane": "cpu", "cpu_workers": 8, "priority": 0, "after": ["other-job"]
    }

``args[0]`` must be a script under ``scripts/``. The job also gets ``GCA_RUN_DIR``
(a directory whose contents are pushed with the results), ``GCA_WORKERS`` and
``GCA_LANE``. The runner only executes commits reachable from the jobs branch and
skips any job that already has a final status on the results branch. Jobs run
concurrently within ``--cpu-budget`` CPU workers and one GPU job at a time; a job
without ``cpu_workers`` runs alone. To stop a running job, push an empty file
``runs/cancel/<job id>`` to the jobs branch; it is noticed within ``--push-every``
seconds. Stop the runner itself with Ctrl+C. The runner exits with code 75 when the
jobs branch holds a newer version of this file; ``scripts/start_runner.ps1`` then
restarts it.
"""

import argparse
import csv
import json
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import psutil  # type: ignore[import-untyped]
except ImportError:  # CPU utilisation is then skipped and noted
    psutil = None

FINAL_STATES = ("done", "failed", "timeout", "cancelled", "rejected")
LOG_TAIL_LINES = 200
MAX_LOG_BYTES = 5_000_000
GIT_TIMEOUT_S = 180
EXIT_UPDATE = 75
CI_PRIORITY = -1000
CI_WORKERS = 2
CI_TIMEOUT_MIN = 60
CI_ARGS = [
    "-m",
    "pytest",
    "tests/",
    "--ignore=tests/test_viewer.py",
    "-q",
    "-p",
    "no:cacheprovider",
]
RETRY_BACKOFF_S = 300
STATE_PUSH_EVERY_S = 1800
# Windows / POSIX exit codes of a process that received Ctrl+C.
CTRL_C_CODES = (3221225786, -1073741510, 130, -2)
UTIL_FIELDS = [
    "hour_utc",
    "samples",
    "cpu_mean_pct",
    "gpu_util_mean_pct",
    "gpu_mem_mean_mb",
    "cpu_lane_busy_frac",
    "gpu_lane_busy_frac",
    "cpu_workers_mean",
    "psutil",
]

GIT_LOCK = threading.RLock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git(repo: Path, *args: str, check: bool = True) -> str:
    # A hung git (network drop, credential prompt) must not freeze the runner:
    # time it out and never let it prompt. Calls are serialised so that
    # concurrent jobs never race on refs or index locks.
    with GIT_LOCK:
        try:
            result = subprocess.run(
                ["git", "-C", str(repo), *args],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=GIT_TIMEOUT_S,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            )
        except subprocess.TimeoutExpired as err:
            raise RuntimeError(f"git {' '.join(args)} timed out") from err
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def default_repo() -> Path:
    """The main checkout, even when this file runs from a linked worktree."""
    here = Path(__file__).resolve().parent
    try:
        common = git(here, "rev-parse", "--path-format=absolute", "--git-common-dir")
        common_path = Path(common.strip())
        if common_path.name == ".git":
            return common_path.parent
    except (RuntimeError, OSError):
        pass
    return here.parent


def runner_version(path: Path) -> Optional[str]:
    """Git blob sha of the runner file as it is on disk."""
    try:
        out = git(path.parent, "hash-object", "--", path.name).strip()
    except (RuntimeError, OSError):
        return None
    return out or None


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


def gpu_sample() -> Optional[dict[str, float]]:
    """Utilisation (%) and memory (MB) of the first GPU, or None."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            return None
        util, used, total = (float(x) for x in out.stdout.splitlines()[0].split(","))
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None
    return {"util": util, "mem_used_mb": used, "mem_total_mb": total}


# ---------------------------------------------------------------- scheduling


def _int(value: Any, default: Optional[int]) -> Optional[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return default


def job_lane(job: dict) -> str:
    return "gpu" if job.get("lane") == "gpu" else "cpu"


def job_workers(job: dict) -> Optional[int]:
    """CPU workers the job reserves, or None for a legacy job that needs the machine."""
    workers = _int(job.get("cpu_workers"), None)
    return workers if workers is not None and workers >= 0 else None


def is_exclusive(job: dict) -> bool:
    return job_workers(job) is None


def job_priority(job: dict) -> int:
    return _int(job.get("priority"), 0) or 0


def sort_key(job: dict) -> tuple[int, str]:
    return (-job_priority(job), str(job.get("id")))


def pick_jobs(queued: list[dict], running: list[dict], budget: int) -> list[dict]:
    """Which ready jobs to start now, in start order.

    Highest priority first (ties by id). Summed ``cpu_workers`` stays within
    ``budget`` and at most one gpu-lane job runs. A job without ``cpu_workers`` runs
    alone: while one runs nothing starts, and while one waits nothing behind it
    starts, so that the machine can drain.
    """
    if any(is_exclusive(j) for j in running):
        return []
    used = sum(job_workers(j) or 0 for j in running)
    gpu_busy = any(job_lane(j) == "gpu" for j in running)
    picked: list[dict] = []
    for job in sorted(queued, key=sort_key):
        if is_exclusive(job):
            if not running and not picked:
                picked.append(job)
            break
        workers = min(job_workers(job) or 0, budget)
        if used + workers > budget:
            continue
        if job_lane(job) == "gpu":
            if gpu_busy:
                continue
            gpu_busy = True
        picked.append(job)
        used += workers
    return picked


def classify(
    jobs: list[dict], states: dict[str, str]
) -> tuple[list[dict], dict[str, list[str]], dict[str, str]]:
    """Split unfinished jobs by their ``after`` dependencies.

    ``states`` maps job id to its final state. Returns (ready jobs,
    {blocked id: ids it waits for}, {id: reason} for jobs to reject).
    """
    ready: list[dict] = []
    blocked: dict[str, list[str]] = {}
    rejected: dict[str, str] = {}
    for job in jobs:
        after = job.get("after") or []
        if not isinstance(after, list):
            ready.append(job)  # validate() rejects the malformed field
            continue
        waiting = []
        for dep in after:
            state = states.get(dep)
            if state == "done":
                continue
            if state is not None:
                rejected[job["id"]] = f"dependency {dep} ended {state}"
                break
            waiting.append(dep)
        else:
            if waiting:
                blocked[job["id"]] = waiting
            else:
                ready.append(job)
    return ready, blocked, rejected


# ------------------------------------------------------------------------ CI


def ci_job_id(sha: str) -> str:
    return f"ci-{sha[:12]}"


def ci_job(sha: str) -> dict[str, Any]:
    return {
        "id": ci_job_id(sha),
        "commit": sha,
        "args": list(CI_ARGS),
        "timeout_min": CI_TIMEOUT_MIN,
        "lane": "cpu",
        "cpu_workers": CI_WORKERS,
        "priority": CI_PRIORITY,
        "internal": "ci",
    }


def ci_next(
    head: Optional[str],
    last_sha: Optional[str],
    last_time: Optional[float],
    pending: Optional[str],
    now_ts: float,
    min_interval_s: float,
) -> Optional[str]:
    """The commit the pending CI job should target, or None for no CI job.

    A waiting job is always retargeted to the newest head. A new one is queued only
    if the head differs from the last tested commit and the last CI job started at
    least ``min_interval_s`` ago.
    """
    if not head:
        return pending
    if pending:
        return head
    if head == last_sha:
        return None
    if last_time is not None and now_ts - last_time < min_interval_s:
        return None
    return head


_COUNT_RE = re.compile(
    r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected)"
)


def parse_pytest_counts(text: str) -> dict[str, int]:
    """Counts from pytest's final summary line (last line that has any)."""
    for line in reversed(text.splitlines()):
        found = _COUNT_RE.findall(line)
        if found and " in " in line:
            counts: dict[str, int] = {}
            for number, word in found:
                key = "errors" if word.startswith("error") else word
                counts[key] = counts.get(key, 0) + int(number)
            return counts
    return {}


def write_ci_files(
    results: "Results", job: dict, state: str, log_text: str, finished_at: str
) -> None:
    sha = job["commit"]
    counts = parse_pytest_counts(log_text)
    folder = results.root / "ci" / sha
    folder.mkdir(parents=True, exist_ok=True)
    last = "".join(log_text.splitlines(keepends=True)[-60:])
    count_line = ", ".join(f"{v} {k}" for k, v in counts.items()) or "no counts found"
    (folder / "summary.txt").write_text(
        f"commit {sha}\nstate {state}\ncounts: {count_line}\n\n{last}", encoding="utf-8"
    )
    latest = {"sha": sha, "state": state, "counts": counts, "time": finished_at}
    (results.root / "ci" / "latest.json").write_text(
        json.dumps(latest, indent=2) + "\n", encoding="utf-8"
    )


def load_ci_latest(results: "Results") -> tuple[Optional[str], Optional[float]]:
    path = results.root / "ci" / "latest.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        stamp = datetime.fromisoformat(data["time"]).timestamp()
        return data["sha"], stamp
    except (OSError, ValueError, KeyError, TypeError):
        return None, None


# --------------------------------------------------------------- utilisation


class HourlyAggregator:
    """Collects samples and emits one CSV row per UTC hour."""

    def __init__(self) -> None:
        self.hour: Optional[str] = None
        self.rows: list[dict] = []

    def add(
        self,
        when: datetime,
        cpu: Optional[float],
        gpu: Optional[dict],
        cpu_busy: bool,
        gpu_busy: bool,
        workers: int,
    ) -> Optional[dict]:
        hour = when.strftime("%Y-%m-%dT%H:00Z")
        finished = None
        if self.hour is not None and hour != self.hour:
            finished = self.flush()
        self.hour = hour
        self.rows.append(
            {
                "cpu": cpu,
                "gpu_util": gpu["util"] if gpu else None,
                "gpu_mem": gpu["mem_used_mb"] if gpu else None,
                "cpu_busy": cpu_busy,
                "gpu_busy": gpu_busy,
                "workers": workers,
            }
        )
        return finished

    def flush(self) -> Optional[dict]:
        if not self.rows or self.hour is None:
            return None
        rows, hour = self.rows, self.hour
        self.rows = []

        def mean(key: str) -> str:
            values = [r[key] for r in rows if r[key] is not None]
            return f"{sum(values) / len(values):.1f}" if values else ""

        def frac(key: str) -> str:
            return f"{sum(1 for r in rows if r[key]) / len(rows):.3f}"

        return {
            "hour_utc": hour,
            "samples": len(rows),
            "cpu_mean_pct": mean("cpu"),
            "gpu_util_mean_pct": mean("gpu_util"),
            "gpu_mem_mean_mb": mean("gpu_mem"),
            "cpu_lane_busy_frac": frac("cpu_busy"),
            "gpu_lane_busy_frac": frac("gpu_busy"),
            "cpu_workers_mean": mean("workers"),
            "psutil": "yes" if psutil is not None else "no",
        }


def append_utilization(results: "Results", row: dict) -> None:
    path = results.root / "runner" / "utilization.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=UTIL_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


# ------------------------------------------------------------------- results


class Results:
    """A worktree on the orphan results branch, written to and pushed by the runner.

    All git operations on it hold one lock, so concurrent jobs share one writer.
    """

    def __init__(self, repo: Path, root: Path, branch: str, remote: str) -> None:
        self.repo, self.root, self.branch, self.remote = repo, root, branch, remote
        self.lock = threading.RLock()
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
        # A runner stopped mid-job leaves unsaved files; keep them rather than
        # failing the rebase.
        with self.lock:
            git(self.root, "add", "-A")
            if git(self.root, "status", "--porcelain").strip():
                git(
                    self.root,
                    "commit",
                    "-q",
                    "-m",
                    "Save files left by a stopped runner",
                )
            if git(self.repo, "ls-remote", "--heads", self.remote, self.branch).strip():
                # Running jobs write into runs/<id>/artifacts here between commits,
                # so the tree can be dirty again by now.
                git(
                    self.root,
                    "pull",
                    "--rebase",
                    "--autostash",
                    "--quiet",
                    self.remote,
                    self.branch,
                )

    def job_dir(self, job_id: str) -> Path:
        path = self.root / "runs" / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def final_state(self, job_id: str) -> Optional[str]:
        status = self.root / "runs" / job_id / "status.json"
        if not status.exists():
            return None
        try:
            state = json.loads(status.read_text(encoding="utf-8")).get("state")
        except (OSError, json.JSONDecodeError):
            return None
        return state if state in FINAL_STATES else None

    def commit_and_push(self, message: str) -> None:
        with self.lock:
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
                    try:
                        self.pull()
                    except RuntimeError as pull_err:
                        print(f"pull failed ({pull_err})", file=sys.stderr)
            print(
                "giving up on this push; will retry at the next update", file=sys.stderr
            )


# ---------------------------------------------------------------------- jobs


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


def cancelled_ids(repo: Path, ref: str) -> set[str]:
    out = git(repo, "ls-tree", "--name-only", f"{ref}:runs/cancel", check=False)
    return {Path(n).name for n in out.split()}


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
        if not isinstance(job, dict):
            continue
        job.setdefault("id", name[: -len(".json")])
        job.pop("internal", None)  # reserved for jobs the runner creates itself
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
    if job.get("internal") != "ci":
        script = args[0].replace("\\", "/")
        if not script.startswith("scripts/") or ".." in script.split("/"):
            return "args[0] must be a script under scripts/"
    if job.get("lane", "cpu") not in ("cpu", "gpu"):
        return 'lane must be "cpu" or "gpu"'
    if "cpu_workers" in job and (_int(job["cpu_workers"], None) in (None,)):
        return "cpu_workers must be an int"
    if "cpu_workers" in job and job["cpu_workers"] < 0:
        return "cpu_workers must be >= 0"
    if "priority" in job and _int(job["priority"], None) is None:
        return "priority must be an int"
    after = job.get("after", [])
    if not isinstance(after, list) or not all(isinstance(a, str) for a in after):
        return "after must be a list of job ids"
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


def reject_job(results: Results, job: dict, reason: str) -> None:
    out_dir = results.job_dir(job["id"])
    (out_dir / "job.json").write_text(
        json.dumps(job, indent=2) + "\n", encoding="utf-8"
    )
    write_status(
        out_dir / "status.json",
        id=job["id"],
        commit=job.get("commit"),
        queued_args=job.get("args"),
        state="rejected",
        reason=reason,
    )
    results.commit_and_push(f"{job['id']}: rejected")
    print(f"{job['id']}: rejected ({reason})")


def run_job(
    args: argparse.Namespace,
    repo: Path,
    results: Results,
    job: dict,
    interrupt: threading.Event,
) -> None:
    job_id = job["id"]
    reason = validate(repo, args.jobs_ref, job)
    if reason:
        reject_job(results, job, reason)
        return

    out_dir = results.job_dir(job_id)
    (out_dir / "job.json").write_text(
        json.dumps(job, indent=2) + "\n", encoding="utf-8"
    )
    status = {"id": job_id, "commit": job.get("commit"), "queued_args": job.get("args")}
    workers = job_workers(job)

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
        "GCA_WORKERS": str(args.cpu_budget if workers is None else workers),
        "GCA_LANE": job_lane(job),
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
            lane=job_lane(job),
            cpu_workers=workers,
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
            if interrupt.is_set():
                proc.kill()
                state = "interrupted"
                break
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
    if state == "done" and (interrupt.is_set() or returncode in CTRL_C_CODES):
        state = "interrupted"
    if state == "interrupted":
        # Not a final state: the job runs again when the runner restarts.
        push_progress("interrupted", finished_at=now())
        git(repo, "worktree", "remove", "--force", str(work), check=False)
        print(f"{job_id}: interrupted; it will run again on the next start")
        return
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
    finished_at = now()
    if job.get("internal") == "ci":
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
            write_ci_files(results, job, state, text, finished_at)
        except OSError as err:
            print(f"{job_id}: could not write CI summary ({err})", file=sys.stderr)
    push_progress(state, returncode=returncode, finished_at=finished_at)
    git(repo, "worktree", "remove", "--force", str(work), check=False)
    print(f"{job_id}: {state} (exit {returncode})")


class Running:
    def __init__(self, job: dict, thread: threading.Thread, stop: threading.Event):
        self.job, self.thread, self.stop = job, thread, stop
        self.started = time.time()


class Scheduler:
    """Starts jobs as threads, keeps state.json and the utilisation record."""

    def __init__(
        self, args: argparse.Namespace, repo: Path, results: Results, version: Any
    ) -> None:
        self.args, self.repo, self.results, self.version = args, repo, results, version
        self.running: dict[str, Running] = {}
        self.wake = threading.Event()
        self.retry_after: dict[str, float] = {}
        self.idle_since: dict[str, Optional[str]] = {"cpu": now(), "gpu": now()}
        self.agg = HourlyAggregator()
        self.ci_last_sha, self.ci_last_time = load_ci_latest(results)
        self.ci_pending: Optional[str] = None
        self.drain_started: Optional[float] = None
        self.last_state_key = ""
        self.last_state_push = 0.0
        if psutil is not None:
            psutil.cpu_percent(interval=None)  # the first reading is meaningless

    # -- threads
    def _target(self, job: dict, stop: threading.Event) -> None:
        try:
            run_job(self.args, self.repo, self.results, job, stop)
        except Exception:  # keep the runner alive; the job retries after a backoff
            traceback.print_exc()
            self.retry_after[job["id"]] = time.time() + RETRY_BACKOFF_S
        finally:
            self.wake.set()

    def start(self, job: dict) -> None:
        stop = threading.Event()
        thread = threading.Thread(
            target=self._target, args=(job, stop), name=job["id"], daemon=True
        )
        self.running[job["id"]] = Running(job, thread, stop)
        thread.start()

    def reap(self) -> None:
        for job_id in [i for i, r in self.running.items() if not r.thread.is_alive()]:
            del self.running[job_id]

    def stop_all(self, wait_s: float = 120) -> None:
        for r in self.running.values():
            r.stop.set()
        deadline = time.time() + wait_s
        for r in self.running.values():
            r.thread.join(max(0.0, deadline - time.time()))

    # -- records
    def sample(self) -> tuple[Optional[float], Optional[dict]]:
        cpu = psutil.cpu_percent(interval=None) if psutil is not None else None
        gpu = gpu_sample()
        jobs = [r.job for r in self.running.values()]
        exclusive = any(is_exclusive(j) for j in jobs)
        cpu_busy = exclusive or any(job_lane(j) == "cpu" for j in jobs)
        gpu_busy = exclusive or any(job_lane(j) == "gpu" for j in jobs)
        for lane, busy in (("cpu", cpu_busy), ("gpu", gpu_busy)):
            if busy:
                self.idle_since[lane] = None
            elif self.idle_since[lane] is None:
                self.idle_since[lane] = now()
        workers = sum(job_workers(j) or 0 for j in jobs)
        row = self.agg.add(
            datetime.now(timezone.utc), cpu, gpu, cpu_busy, gpu_busy, workers
        )
        if row:
            append_utilization(self.results, row)
            self.results.commit_and_push("runner: utilization")
        return cpu, gpu

    def write_state(
        self, ready: list[dict], blocked: dict[str, list[str]], cpu: Any, gpu: Any
    ) -> None:
        jobs = [r.job for r in self.running.values()]
        state = {
            "version": self.version,
            "host": platform.node(),
            "draining_for_update": self.drain_started is not None,
            "cpu_budget": self.args.cpu_budget,
            "cpu_workers_in_use": sum(job_workers(j) or 0 for j in jobs),
            "running": [
                {
                    "id": r.job["id"],
                    "lane": job_lane(r.job),
                    "cpu_workers": job_workers(r.job),
                    "started_at": datetime.fromtimestamp(
                        r.started, timezone.utc
                    ).isoformat(timespec="seconds"),
                }
                for r in self.running.values()
            ],
            "backlog": {
                "ready": [j["id"] for j in sorted(ready, key=sort_key)],
                "blocked": [
                    {"id": i, "blocked_by": deps} for i, deps in sorted(blocked.items())
                ],
            },
            "idle_since": dict(self.idle_since),
            "psutil": psutil is not None,
        }
        key = json.dumps(state, sort_keys=True)
        state.update(updated_at=now(), cpu_percent=cpu, gpu=gpu)
        path = self.results.root / "runner" / "state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        if key != self.last_state_key or time.time() - self.last_state_push > (
            STATE_PUSH_EVERY_S
        ):
            self.results.commit_and_push("runner: state")
            self.last_state_key, self.last_state_push = key, time.time()

    # -- one scheduling pass
    def update_requested(self) -> bool:
        if self.args.no_update or self.args.once or not self.version:
            return False
        theirs = git(
            self.repo,
            "rev-parse",
            f"{self.args.jobs_ref}:scripts/gpu_runner.py",
            check=False,
        ).strip()
        return bool(theirs) and theirs != self.version

    def step(self) -> bool:
        """One pass. Returns True when the runner should exit (update or --once)."""
        args = self.args
        self.reap()
        cpu, gpu = self.sample()
        try:
            git(self.repo, "fetch", "-q", args.remote, args.jobs_branch)
        except RuntimeError as err:
            print(f"sync failed ({err}); retrying in {args.poll} s", file=sys.stderr)
            return False
        try:
            self.results.pull()
        except RuntimeError as err:
            # Final states are read from the local results tree, which this runner
            # writes itself, so scheduling can go on; the next push retries the sync.
            print(f"results sync failed ({err}); scheduling anyway", file=sys.stderr)

        if self.update_requested() and self.drain_started is None:
            self.drain_started = time.time()
            print("newer gpu_runner.py on the jobs branch; draining for update")
        if self.drain_started is not None:
            waited = (time.time() - self.drain_started) / 60
            if self.running and waited > args.update_drain_min:
                print("drain limit reached; interrupting running jobs")
                self.stop_all()
                self.reap()
            if not self.running:
                self.flush_utilization()
                sys.exit(EXIT_UPDATE)

        jobs = queued_jobs(self.repo, args.jobs_ref)
        cancelled = cancelled_ids(self.repo, args.jobs_ref)
        if args.ci and not args.once:
            self.plan_ci()
            if self.ci_pending and ci_job_id(self.ci_pending) not in {
                j["id"] for j in jobs
            }:
                jobs.append(ci_job(self.ci_pending))

        states: dict[str, str] = {}
        todo: list[dict] = []
        for job in jobs:
            final = self.results.final_state(job["id"])
            if final is not None:
                states[job["id"]] = final
            elif job["id"] in cancelled:
                states[job["id"]] = "cancelled"
            elif job["id"] not in self.running:
                todo.append(job)
        ready, blocked, rejected = classify(todo, states)
        for job in todo:
            if job["id"] in rejected:
                reject_job(self.results, job, rejected[job["id"]])
        t = time.time()
        ready = [j for j in ready if self.retry_after.get(j["id"], 0) <= t]

        picked = [] if self.drain_started else self.pick(ready)
        for job in picked:
            if job.get("internal") == "ci":
                self.ci_last_sha, self.ci_last_time = job["commit"], time.time()
                self.ci_pending = None
            self.start(job)
        started_ids = {j["id"] for j in picked}
        self.write_state(
            [j for j in ready if j["id"] not in started_ids], blocked, cpu, gpu
        )
        return args.once and not self.running and not picked

    def pick(self, ready: list[dict]) -> list[dict]:
        return pick_jobs(
            ready, [r.job for r in self.running.values()], self.args.cpu_budget
        )

    def plan_ci(self) -> None:
        head = git(self.repo, "rev-parse", self.args.jobs_ref, check=False).strip()
        if head and self.results.final_state(ci_job_id(head)) is not None:
            self.ci_last_sha = head  # already tested (for instance before a restart)
            if self.ci_pending == head:
                self.ci_pending = None
        self.ci_pending = ci_next(
            head or None,
            self.ci_last_sha,
            self.ci_last_time,
            self.ci_pending,
            time.time(),
            self.args.ci_min_interval * 60,
        )

    def flush_utilization(self) -> None:
        row = self.agg.flush()
        if row:
            append_utilization(self.results, row)
            try:
                self.results.commit_and_push("runner: utilization")
            except RuntimeError as err:
                print(f"could not push utilization ({err})", file=sys.stderr)

    def run(self) -> None:
        try:
            while True:
                if self.step():
                    return
                for _ in range(max(1, self.args.poll)):
                    if self.wake.wait(1):
                        break
                self.wake.clear()
        except KeyboardInterrupt:
            self.stop_all()
            self.flush_utilization()
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--repo",
        type=Path,
        default=None,
        help="Main checkout used for job worktrees and results (default: detected)",
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
    parser.add_argument(
        "--cpu-budget",
        type=int,
        default=24,
        help="Max summed cpu_workers of concurrently running jobs",
    )
    parser.add_argument(
        "--ci",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Test every new head of the jobs branch (low priority)",
    )
    parser.add_argument(
        "--ci-min-interval",
        type=float,
        default=30,
        help="Minutes between CI runs",
    )
    parser.add_argument(
        "--update-drain-min",
        type=float,
        default=120,
        help="Minutes to wait for running jobs before an update restart",
    )
    parser.add_argument(
        "--no-update",
        action="store_true",
        help="Never exit for a newer gpu_runner.py on the jobs branch",
    )
    parser.add_argument("--once", action="store_true", help="Run pending jobs and exit")
    args = parser.parse_args()

    repo = (args.repo or default_repo()).resolve()
    args.jobs_ref = f"{args.remote}/{args.jobs_branch}"
    results_dir = args.results_dir or repo.parent / f"{repo.name}-gpu-results"
    while True:
        # After a reboot the network may come up after the runner does.
        try:
            git(repo, "fetch", "-q", args.remote, args.jobs_branch)
            results = Results(
                repo, results_dir.resolve(), args.results_branch, args.remote
            )
            break
        except RuntimeError as err:
            print(f"startup sync failed ({err}); retrying in {args.poll} s")
            time.sleep(args.poll)
    version = runner_version(Path(__file__).resolve())
    print(f"watching {args.jobs_ref}:runs/queue; results -> {args.results_branch}")
    print(f"runner {version or 'unknown version'}; cpu budget {args.cpu_budget}")
    print(f"GPU: {gpu_stats() or 'nvidia-smi not found'}")
    if psutil is None:
        print("psutil not installed: CPU utilisation will not be recorded")

    Scheduler(args, repo, results, version).run()


if __name__ == "__main__":
    main()
