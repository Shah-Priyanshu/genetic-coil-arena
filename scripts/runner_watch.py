"""Watch the PC runner and exit when it needs attention.

Reads the results worktree the runner writes into (``runner/state.json``,
``runs/<id>/status.json``, ``ci/<sha>/summary.txt``), prints every job start and end
(appended to a log as well), and exits with a final ``EXIT <reason>`` line when:

- ``pc-idle`` (code 2): both lanes have been idle for ``--idle-min``;
- ``lane-idle`` (code 2): a lane that was busy during this watch has been idle that long;
- ``failure`` (code 3): a job ended in a state other than ``done``, or a CI run reported
  failed tests or errors;
- ``stale`` (code 4): ``state.json`` has not been updated for ``--stale-min`` (runner down).

Meant to run in the background of a Claude session, so its exit wakes the session:

    python scripts/runner_watch.py --results D:/Projects/genetic-coil-arena-gpu-results
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

LANES = ("cpu", "gpu")
EXIT_CODES = {"pc-idle": 2, "lane-idle": 2, "failure": 3, "stale": 4}


def parse_time(stamp: Optional[str]) -> Optional[float]:
    if not stamp:
        return None
    try:
        return datetime.fromisoformat(stamp).timestamp()
    except ValueError:
        return None


def check(
    state: dict,
    now: float,
    was_busy: dict[str, bool],
    idle_min: float,
    stale_min: float,
) -> Optional[tuple[str, str]]:
    """The reason to stop watching, with details, or None while all is well."""
    updated = parse_time(state.get("updated_at"))
    if updated is None or now - updated > 60 * stale_min:
        return "stale", f"state.json last updated {state.get('updated_at')}"
    # a running job wins over idle_since, which the runner refreshes only when it samples
    busy = {r.get("lane", "cpu") for r in state.get("running", [])}
    idle_for = {}
    for lane in LANES:
        since = parse_time((state.get("idle_since") or {}).get(lane))
        lane_idle = since is not None and lane not in busy
        idle_for[lane] = (now - since) / 60 if lane_idle else 0.0
    idle = [lane for lane in LANES if idle_for[lane] >= idle_min]
    if len(idle) == len(LANES):
        return "pc-idle", ", ".join(
            f"{k} idle {v:.0f} min" for k, v in idle_for.items()
        )
    for lane in idle:
        if was_busy.get(lane):
            return "lane-idle", f"{lane} idle {idle_for[lane]:.0f} min"
    return None


_COUNTS = re.compile(r"^counts: (.*)$", re.M)


def ci_problem(summary: str) -> Optional[str]:
    """Why a CI summary (``ci/<sha>/summary.txt``) is not a clean pass, or None."""
    if not re.search(r"^state done$", summary, re.M):
        state = re.search(r"^state (\S+)$", summary, re.M)
        return f"state {state.group(1) if state else 'unknown'}"
    found = _COUNTS.search(summary)
    counts = found.group(1) if found else "no counts found"
    if "no counts found" in counts or re.search(r"\d+ (failed|errors)", counts):
        return counts
    return None


class Watch:
    def __init__(
        self,
        results: Path,
        log: Optional[Path] = None,
        idle_min: float = 10.0,
        stale_min: float = 20.0,
    ):
        self.results = results
        self.log = log
        self.idle_min, self.stale_min = idle_min, stale_min
        self.started = time.time()
        self.reported: set[str] = set()
        self.running: dict[str, str] = {}
        self.was_busy = dict.fromkeys(LANES, False)

    def say(self, line: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"{stamp} {line}", flush=True)
        if self.log is not None:
            with self.log.open("a", encoding="utf-8") as fh:
                fh.write(f"{stamp} {line}\n")

    def state(self) -> dict:
        path = self.results / "runner" / "state.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def ended(self) -> list[dict]:
        """Jobs that finished since the watch started and are not yet reported."""
        out = []
        for path in self.results.glob("runs/*/status.json"):
            if path.parent.name in self.reported:
                continue
            try:
                if path.stat().st_mtime < self.started - 120:
                    continue
                status = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            finished = parse_time(status.get("finished_at"))
            if finished is not None and finished >= self.started - 60:
                out.append(status)
        return sorted(out, key=lambda s: s.get("finished_at", ""))

    def poll(self) -> Optional[tuple[str, str]]:
        state = self.state()
        now = time.time()
        running = {r["id"]: r.get("lane", "cpu") for r in state.get("running", [])}
        for job_id, lane in running.items():
            self.was_busy[lane] = True
            if job_id not in self.running:
                self.say(f"start {job_id} ({lane})")
        self.running = running
        problem = None
        for status in self.ended():
            job_id = status.get("id") or "?"
            self.reported.add(job_id)
            line = f"end {job_id} {status.get('state')} {status.get('elapsed_min')} min"
            bad = status.get("state") != "done"
            if job_id.startswith("ci-"):
                summary = (
                    self.results / "ci" / str(status.get("commit")) / "summary.txt"
                )
                try:
                    why = ci_problem(summary.read_text(encoding="utf-8"))
                except OSError:
                    why = "no CI summary"
                line += f" | tests: {why or 'pass'}"
                bad = bad or why is not None
            self.say(line)
            if bad and problem is None:
                problem = ("failure", line)
        if problem is not None:
            return problem
        return check(state, now, self.was_busy, self.idle_min, self.stale_min)

    def run(self, poll_s: float) -> int:
        self.say(f"watching {self.results}")
        while True:
            outcome = self.poll()
            if outcome is not None:
                reason, details = outcome
                backlog = self.state().get("backlog", {})
                self.say(
                    f"EXIT {reason}: {details}; running {sorted(self.running)}; "
                    f"ready {backlog.get('ready', [])}; "
                    f"blocked {[b.get('id') for b in backlog.get('blocked', [])]}"
                )
                return EXIT_CODES[reason]
            time.sleep(poll_s)


def main(argv: Optional[list[str]] = None) -> int:
    default_log = (
        Path(os.environ.get("LOCALAPPDATA", Path.home()))
        / "genetic_coil_arena"
        / "runner_watch.log"
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    here = Path(__file__).resolve().parents[1]
    ap.add_argument(
        "--results", type=Path, default=here.parent / "genetic-coil-arena-gpu-results"
    )
    ap.add_argument("--idle-min", type=float, default=10.0)
    ap.add_argument("--stale-min", type=float, default=20.0)
    ap.add_argument("--poll", type=float, default=60.0, help="Seconds between checks")
    ap.add_argument("--log", type=Path, default=default_log)
    args = ap.parse_args(argv)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    watch = Watch(args.results, args.log, args.idle_min, args.stale_min)
    return watch.run(args.poll)


if __name__ == "__main__":
    sys.exit(main())
