# Running experiments on the GPU machine

The cloud session can't reach the GPU PC directly, so the two talk through git:

1. A job file is pushed to `runs/queue/<id>.json` on the jobs branch (default
   `claude/vibrant-hawking-zodb5a`). Each job names an exact commit and a script under
   `scripts/`.
2. `scripts/gpu_runner.py`, running on the PC, polls that branch. It runs each new job
   in a fresh, detached worktree of the job's commit, one job at a time.
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

## Start the runner

```powershell
cd D:\Projects\genetic-coil-arena
$env:NUMBA_CACHE_DIR = "C:\ncc"
D:\venvs\gca\Scripts\python scripts\gpu_runner.py --python D:\venvs\gca\Scripts\python.exe
```

Leave it running; Ctrl+C stops it. The runner keeps a results worktree next to the repo
(`..\genetic-coil-arena-gpu-results`). Restarting is safe, because jobs that already
finished are skipped.

Options:

| Option | Effect |
|---|---|
| `--once` | Run any pending jobs, then exit |
| `--poll 60` | Seconds between checks for new jobs |
| `--push-every 300` | Seconds between progress pushes |
| `--jobs-branch` | Branch to read jobs from |

## Job file

```json
{
  "id": "r6-gpu-001",
  "commit": "<full sha on the jobs branch>",
  "args": ["scripts/learned_protocol.py", "--scale", "gpu"],
  "timeout_min": 240,
  "env": {"SOME_FLAG": "1"},
  "outputs": ["docs/results/learned_protocol.md"]
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
| `interrupted` | The runner was stopped with Ctrl+C; the job runs again on the next start |
| `cancelled` | Stopped because `runs/cancel/<id>` was pushed to the jobs branch |
| `rejected` | Never run; `reason` says why (bad script path, or a commit not on the jobs branch) |
