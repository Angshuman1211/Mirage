# Job B, the six sigma-arm retrains, for the CUDA box.
#
# Lives in the repo rather than a temp scratchpad on purpose. The first two
# attempts were launched from a session-scoped temp directory and both died
# about a minute in with nothing in the log, which is what a process group
# getting cleaned up looks like. A script under version control that is run
# detached cannot be killed that way.
#
# Three changes from the bash version:
#   1. PYTHONUNBUFFERED, so a traceback reaches the log before a kill. Python
#      block-buffers stdout when it is redirected, so "no error in the log" was
#      never evidence that no error happened.
#   2. All streams merged per arm into its own file, so one dead arm does not
#      cost the others' output.
#   3. h9_rad_sigma_s42 runs FIRST. It is the arm that tests the 1.12-1.16
#      prediction against the published 1.3729, so it is the only one whose
#      result changes a decision. If only one arm ever finishes, make it that one.
#
# Run detached so it outlives any agent session or terminal:
#   Start-Process powershell -ArgumentList '-NoExit','-File','scripts\run_jobB.ps1'
#
# Or in the foreground:
#   powershell -File scripts\run_jobB.ps1

$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repo
$env:PYTHONUNBUFFERED = "1"

$logdir = Join-Path $repo "logs\jobB"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null

# rad first: it is the one that settles the sigma column
$arms = @(
    "h9_rad_sigma_s42",
    "h9_rad_sigma_s43",
    "h9_rad_sigma_s44",
    "h9_abc_sigma_s42",
    "h9_abc_sigma_s43",
    "h9_abc_sigma_s44"
)

Write-Host "Job B starting $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  repo $repo"

foreach ($arm in $arms) {
    $dir  = Join-Path $repo "configs\$arm"
    $best = Join-Path $dir "model__best.pt"
    $log  = Join-Path $logdir "$arm.log"

    if (Test-Path $best) {
        Write-Host "SKIP $arm, model__best.pt already present"
        continue
    }
    if (-not (Test-Path (Join-Path $dir "config.yaml"))) {
        Write-Host "SKIP $arm, no config.yaml"
        continue
    }

    Write-Host "=== $arm  start $(Get-Date -Format 'HH:mm:ss') ==="
    "=== $arm start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File -FilePath $log -Encoding utf8

    # *>&1 merges stderr and every other stream into the log
    python -u scripts\train.py --experiment-dir "configs\$arm" *>&1 |
        Tee-Object -FilePath $log -Append

    if (Test-Path $best) {
        Write-Host "=== $arm  DONE $(Get-Date -Format 'HH:mm:ss') ==="
    } else {
        Write-Host "=== $arm  NO CHECKPOINT, see $log ==="
        Write-Host "    last 20 lines:"
        Get-Content $log -Tail 20 | ForEach-Object { Write-Host "    $_" }
    }
}

Write-Host "Job B finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "Health check each arm against its reference with:"
Write-Host "  python scripts\check_run_health.py"
