# Running experiments on the GPU machine

The cloud session can't reach the GPU PC directly, so the two talk through git:

1. A job file is pushed to `runs/queue/<id>.json` on the jobs branch (default
   `claude/vibrant-hawking-zodb5a`). Each job names an exact commit and a script under
   `scripts/`.
2. `scripts/gpu_runner.py`, running on the PC, polls that branch. It runs each new job
   in a fresh, detached worktree of the job's commit. Jobs run concurrently in two lanes
   (see "Lanes and resources").
3. While the job runs, the runner pushes `status.json` (elapsed time, GPU usage and
   memory) and the last 200 log lines to the orphan branch `gpu-results`, every 5
   minutes. When the job ends, it pushes the full log, the job's `outputs` and anything
   the job wrote to `$GCA_RUN_DIR`.

The runner never touches your working checkout and uses no Claude usage. It only runs
commits that are on the jobs branch, and only scripts under `scripts/`. It does run
whatever code is on that branch, so only point it at a branch you trust.

## One-time setup (Windows, PowerShell)

```powershell
cd D:\Projects\genetic-coil-arena
git fetch origin
git checkout claude/vibrant-hawking-zodb5a

py -3.13 -m venv D:\venvs\gca   # any Python torch supports; `py -0` lists them
D:\venvs\gca\Scripts\python -m pip install --upgrade pip
D:\venvs\gca\Scripts\python -m pip install -e ".[dev]"
D:\venvs\gca\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Use a CUDA build of torch that matches your driver (`nvidia-smi` shows the highest CUDA
version it supports). Python 3.10–3.13 all work with current torch.

`git push` must work without a prompt. If you've pushed from this folder before, it
already does.

## Start the runner (supervisor, recommended)

```powershell
cd D:\Projects\genetic-coil-arena
.\scripts\start_runner.ps1 -Python D:\venvs\gca\Scripts\python.exe
```

The supervisor keeps a dedicated worktree `..\genetic-coil-arena-runner` at the jobs-branch
head and runs `gpu_runner.py` from it. When the jobs branch gets a newer
`scripts/gpu_runner.py`, the runner stops starting jobs, lets running ones finish (at most
`--update-drain-min`, default 120; then they are marked `interrupted` and rerun) and exits
with code 75. The supervisor then updates the worktree and restarts it, so a runner upgrade
is just a push. After any other exit it waits 60 s and restarts. Ctrl+C stops everything.
Extra arguments go to the runner (`.\scripts\start_runner.ps1 --cpu-budget 20`).

To switch from a runner started by hand: Ctrl+C it (a running job becomes `interrupted` and
reruns), then start the supervisor. `pip install psutil` in the job venv to record CPU use.

Without the supervisor, run `python scripts\gpu_runner.py --python ...` from the main
checkout (add `--no-update` to disable the exit-75 behaviour). The runner always uses the main
checkout (`--repo`, detected even when run from the runner worktree) for job worktrees, and
keeps its results worktree next to it (`..\genetic-coil-arena-gpu-results`). Restarting is
safe, because jobs that already finished are skipped.

| Option | Effect |
|---|---|
| `--once` | Run any pending jobs, then exit (no CI, no update check) |
| `--poll 60` | Seconds between checks for new jobs |
| `--push-every 300` | Seconds between progress pushes |
| `--jobs-branch` | Branch to read jobs from |
| `--cpu-budget 24` | Max summed `cpu_workers` of running jobs |
| `--ci` / `--no-ci` | Automatic CI of the jobs-branch head (default on) |
| `--ci-min-interval 30` | Minutes between CI runs |
| `--update-drain-min 120` | Max wait for running jobs before an update restart |

## Lanes and resources

Optional job fields (old job files still work):

| Field | Meaning |
|---|---|
| `lane` | `"cpu"` (default) or `"gpu"`. At most one gpu-lane job runs at a time |
| `cpu_workers` | CPU workers the job uses; passed as `GCA_WORKERS` (`GCA_LANE` gets the lane). Jobs start while the summed workers of running jobs stay within `--cpu-budget` |
| `priority` | Int, higher first, default 0; ties by id |
| `after` | Job ids that must be `done` first. If one ended in any other final state the job is `rejected` with a reason |

A job **without `cpu_workers` runs exclusively**: nothing starts while it runs, and while it
waits nothing of lower priority starts, so the machine drains. Give every new job
`cpu_workers` (use `0` for a GPU job that needs no CPU workers).

## Automatic CI

With `--ci`, when the jobs-branch head changes the runner queues a job `ci-<sha12>`
(priority -1000, 2 workers, 60 min timeout, at most one per `--ci-min-interval`, always for
the latest head). It runs `python -m pytest tests/ --ignore=tests/test_viewer.py -q -p no:cacheprovider`
in a worktree of that commit and writes `ci/<sha>/summary.txt` (last 60 lines and counts) and
`ci/latest.json` (sha, state, counts, time) to `gpu-results`.

## Utilization and state

Each poll samples CPU % (psutil, if installed), GPU utilisation and memory (nvidia-smi) and
which lanes are busy. `runner/utilization.csv` on `gpu-results` gets one aggregate row per
UTC hour (goal G0). `runner/state.json` holds the runner version (blob sha of
`gpu_runner.py`), running jobs, the backlog (ready jobs; blocked jobs and what blocks them)
and idle-since per lane.

## Queueing jobs

`scripts/queue_job.py` writes the job files; commit and push them yourself. The commit
defaults to `HEAD`, which must already be on the jobs branch.

```powershell
python scripts\queue_job.py r8-{seed} --workers 4 --priority 1 --grid seed=1,2,3 `
    --outputs docs/results/r8_{seed}.md -- scripts/code_formation.py --seed {seed}
git add runs/queue; git commit -m "queue r8"; git push
```

Flags: `--lane`, `--workers`, `--priority`, `--after id,id`, `--timeout-min`, `--outputs`
(repeatable), `--env K=V` (repeatable), `--commit`, `--force`. `--grid name=v1,v2` expands
`{name}` in the id, args, outputs and `--after` into one job per combination.

## Job file

```json
{
  "id": "r6-gpu-001",
  "commit": "<full sha on the jobs branch>",
  "args": ["scripts/learned_protocol.py", "--scale", "gpu"],
  "timeout_min": 240,
  "env": {"SOME_FLAG": "1"},
  "outputs": ["docs/results/learned_protocol.md"],
  "lane": "cpu", "cpu_workers": 8, "priority": 0, "after": ["other-job"]
}
```

## Reading results

```bash
git fetch origin gpu-results
git show origin/gpu-results:runs/<id>/status.json
git show origin/gpu-results:runs/<id>/log_tail.txt
```

The status `state` is one of:

| State | Meaning |
|---|---|
| `running` | In progress |
| `done` | Finished with exit code 0 |
| `failed` | Exited with a non-zero code |
| `timeout` | Killed after `timeout_min` |
| `interrupted` | The runner was stopped (Ctrl+C or an update restart); the job runs again on the next start |
| `cancelled` | Stopped because `runs/cancel/<id>` was pushed to the jobs branch |
| `rejected` | Never run; `reason` says why (bad script path or field, a commit not on the jobs branch, or a failed dependency) |
