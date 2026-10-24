# Supervisor for scripts/gpu_runner.py: keeps a dedicated worktree at the jobs-branch
# head, runs the runner from it, and restarts it after an update (exit code 75) or a crash.
# Ctrl+C stops everything. Extra arguments go to gpu_runner.py.
#
#   .\scripts\start_runner.ps1
#   .\scripts\start_runner.ps1 -Python D:\venvs\gca\Scripts\python.exe --cpu-budget 20
param(
    [string]$Python = "D:\venvs\gca\Scripts\python.exe",
    [string]$Repo = (Split-Path -Parent $PSScriptRoot),
    [string]$JobsBranch = "claude/vibrant-hawking-zodb5a",
    [string]$Remote = "origin",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunnerArgs
)

$ErrorActionPreference = "Continue"
$Repo = (Resolve-Path $Repo).Path
$Worktree = Join-Path (Split-Path -Parent $Repo) "$(Split-Path -Leaf $Repo)-runner"
if (-not $env:NUMBA_CACHE_DIR) { $env:NUMBA_CACHE_DIR = "C:\ncc" }
$env:GIT_TERMINAL_PROMPT = "0"

function Update-Worktree {
    git -C $Repo fetch -q $Remote $JobsBranch 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Host "fetch failed; using the existing runner worktree" }
    $ref = "$Remote/$JobsBranch"
    if (-not (Test-Path (Join-Path $Worktree ".git"))) {
        git -C $Repo worktree prune 2>&1 | Out-Null
        git -C $Repo worktree add --detach --force $Worktree $ref 2>&1 | Out-Null
    } else {
        git -C $Worktree checkout -q -f --detach $ref 2>&1 | Out-Null
    }
    return (Test-Path (Join-Path $Worktree "scripts\gpu_runner.py"))
}

while ($true) {
    if (Update-Worktree) {
        Write-Host "$(Get-Date -Format s) starting runner from $Worktree"
        & $Python (Join-Path $Worktree "scripts\gpu_runner.py") --repo $Repo --python $Python `
            --remote $Remote --jobs-branch $JobsBranch @RunnerArgs
        $code = $LASTEXITCODE
    } else {
        Write-Host "runner worktree is missing scripts\gpu_runner.py"
        $code = 1
    }
    if ($code -eq 75) {
        Write-Host "$(Get-Date -Format s) runner updated; restarting"
        continue
    }
    Write-Host "$(Get-Date -Format s) runner exited with $code; restarting in 60 s"
    Start-Sleep -Seconds 60
}
