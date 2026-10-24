"""Write job files for the runner (``scripts/gpu_runner.py``) into ``runs/queue/``.

    python scripts/queue_job.py r8-sweep-{seed} --workers 4 --grid seed=1,2,3 \\
        --outputs docs/results/r8_{seed}.md -- scripts/code_formation.py --seed {seed}

The job is pinned to a commit (default HEAD) that must already be pushed to the jobs
branch. Nothing is committed or pushed: add, commit and push the printed files
yourself. ``--grid name=v1,v2`` expands ``{name}`` in the id, the script args,
``--outputs`` and ``--after`` into one job per combination.
"""

import argparse
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def parse_grid(items: list[str]) -> dict[str, list[str]]:
    grid: dict[str, list[str]] = {}
    for item in items:
        name, sep, values = item.partition("=")
        parts = [v for v in values.split(",") if v != ""]
        if not sep or not re.fullmatch(r"\w+", name) or not parts:
            raise ValueError(f"bad --grid {item!r}; use name=v1,v2,...")
        if name in grid:
            raise ValueError(f"--grid {name} given twice")
        grid[name] = parts
    return grid


def substitute(text: str, combo: dict[str, str]) -> str:
    for name, value in combo.items():
        text = text.replace("{" + name + "}", value)
    return text


def expand_grid(
    job_id: str, texts: list[str], grid: dict[str, list[str]]
) -> list[tuple[str, dict[str, str]]]:
    """(expanded id, combination) for every grid point; checks the id stays unique."""
    for name in grid:
        if "{" + name + "}" not in job_id:
            raise ValueError(f"id must contain {{{name}}} so each job gets its own id")
        if not any("{" + name + "}" in t for t in [job_id, *texts]):
            raise ValueError(f"grid name {name} is not used")
    combos = [dict(zip(grid, vals)) for vals in itertools.product(*grid.values())]
    return [(substitute(job_id, c), c) for c in combos]


def build_jobs(
    job_id: str,
    commit: str,
    script_args: list[str],
    grid: dict[str, list[str]],
    lane: str = "cpu",
    workers: Optional[int] = None,
    priority: int = 0,
    after: Optional[list[str]] = None,
    timeout_min: Optional[float] = None,
    outputs: Optional[list[str]] = None,
    env: Optional[dict[str, str]] = None,
) -> list[dict]:
    if not script_args:
        raise ValueError("give the script and its args after --")
    script = script_args[0].replace("\\", "/")
    if not script.startswith("scripts/") or ".." in script.split("/"):
        raise ValueError("the script must be under scripts/")
    if lane not in ("cpu", "gpu"):
        raise ValueError('lane must be "cpu" or "gpu"')
    if workers is not None and workers < 0:
        raise ValueError("workers must be >= 0")
    after = after or []
    outputs = outputs or []
    texts = [*script_args, *after, *outputs]
    jobs = []
    for expanded_id, combo in expand_grid(job_id, texts, grid):
        if not ID_RE.match(expanded_id):
            raise ValueError(f"bad job id {expanded_id!r}; use letters, digits, . _ -")
        job: dict = {
            "id": expanded_id,
            "commit": commit,
            "args": [substitute(a, combo) for a in script_args],
            "lane": lane,
        }
        if workers is not None:
            job["cpu_workers"] = workers
        if priority:
            job["priority"] = priority
        if after:
            job["after"] = [substitute(a, combo) for a in after]
        if timeout_min is not None:
            job["timeout_min"] = timeout_min
        if outputs:
            job["outputs"] = [substitute(o, combo) for o in outputs]
        if env:
            job["env"] = dict(env)
        jobs.append(job)
    ids = [j["id"] for j in jobs]
    if len(set(ids)) != len(ids):
        raise ValueError("grid produced duplicate ids")
    return jobs


def git_out(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True
    )


def resolve_commit(repo: Path, rev: str, ref: str) -> str:
    """Full sha of ``rev``, which must be an ancestor of ``ref`` (pushed)."""
    out = git_out(repo, "rev-parse", "--verify", rev + "^{commit}")
    if out.returncode != 0:
        raise ValueError(f"unknown commit {rev!r}")
    sha = out.stdout.strip()
    if git_out(repo, "merge-base", "--is-ancestor", sha, ref).returncode != 0:
        raise ValueError(
            f"{rev} ({sha[:10]}) is not on {ref}; push it to the jobs branch first"
        )
    return sha


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    script_args: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, script_args = argv[:cut], argv[cut + 1 :]
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        usage="queue_job.py ID [options] -- script args...",
    )
    parser.add_argument("id")
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--lane", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--workers", type=int, help="cpu_workers; omit to run alone")
    parser.add_argument("--priority", type=int, default=0)
    parser.add_argument("--after", action="append", default=[], help="id,id,...")
    parser.add_argument("--timeout-min", type=float)
    parser.add_argument("--outputs", action="append", default=[])
    parser.add_argument("--env", action="append", default=[], metavar="K=V")
    parser.add_argument("--grid", action="append", default=[], metavar="name=v1,v2")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--jobs-branch", default="claude/vibrant-hawking-zodb5a")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--force", action="store_true", help="overwrite job files")
    args = parser.parse_args(argv)

    try:
        top = git_out(args.repo, "rev-parse", "--show-toplevel")
        if top.returncode != 0:
            raise ValueError(f"{args.repo} is not a git repository")
        repo = Path(top.stdout.strip())
        commit = resolve_commit(repo, args.commit, f"{args.remote}/{args.jobs_branch}")
        env = {}
        for item in args.env:
            key, sep, value = item.partition("=")
            if not sep or not key:
                raise ValueError(f"bad --env {item!r}; use K=V")
            env[key] = value
        after = [a for item in args.after for a in item.split(",") if a]
        jobs = build_jobs(
            args.id,
            commit,
            script_args,
            parse_grid(args.grid),
            lane=args.lane,
            workers=args.workers,
            priority=args.priority,
            after=after,
            timeout_min=args.timeout_min,
            outputs=args.outputs,
            env=env,
        )
        queue_dir = repo / "runs" / "queue"
        existing = [j["id"] for j in jobs if (queue_dir / f"{j['id']}.json").exists()]
        if existing and not args.force:
            raise ValueError(f"already queued: {', '.join(existing)} (--force)")
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    queue_dir.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        path = queue_dir / f"{job['id']}.json"
        path.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
        print(path)
    print(f"{len(jobs)} job file(s) written; git add, commit and push them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
