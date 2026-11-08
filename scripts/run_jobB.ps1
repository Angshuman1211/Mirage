# Job B, the six sigma-arm retrains, for the CUDA box.
#
# Lives in the repo rather than a temp scratchpad on purpose. The first two
# attempts were launched from a session-scoped temp directory and both died
# about a minute in with nothing in the log, which is what a process group
# getting cleaned up looks like. A script under version control that is run
# detached cannot be killed that way.
#
#   Start-Process powershell -ArgumentList '-NoExit','-File','scripts\run_jobB.ps1'
#
# If `python` on PATH is not the env that has mirage and fm4ar installed, pass
# the right interpreter. Find it with `conda env list`, then:
#
#   powershell -File scripts\run_jobB.ps1 -Python "C:\Users\angsh\miniconda3\envs\mirage\python.exe"
#
# WHY THE PREFLIGHT EXISTS. The 16:50 run failed all six arms in one second with
# ModuleNotFoundError because PATH resolved to the wrong interpreter and nothing
# checked before launching. A script that cannot train should say so once, not
# six times. PYTHONPATH is also set to the repo root here, because
# `python scripts\train.py` puts scripts\ on sys.path and NOT the repo root, so
# `import mirage` needs either an editable install or this.

param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repo

$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = $repo          # so import mirage works without an editable install
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$logdir = Join-Path $repo "logs\jobB"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null

Write-Host "Job B $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "  repo        $repo"
Write-Host "  interpreter $Python"
Write-Host ""

# ---- preflight. Nothing trains until this passes. ----------------------------
$probe = @'
import sys
ok = True
try:
    import torch
    print(f"  torch        {torch.__version__}")
    print(f"  cuda         {torch.cuda.is_available()}", end="")
    if torch.cuda.is_available():
        print(f"  {torch.cuda.get_device_name(0)}")
    else:
        print("  <-- no GPU visible"); ok = False
except Exception as e:
    print(f"  torch        FAIL {e}"); ok = False
for m in ("mirage", "fm4ar"):
    try:
        mod = __import__(m)
        print(f"  {m:<12} {getattr(mod, '__file__', '?')}")
    except Exception as e:
        print(f"  {m:<12} FAIL {e}"); ok = False
print(f"  python       {sys.executable}")
sys.exit(0 if ok else 1)
'@

Write-Host "Preflight:"
$probeFile = Join-Path $logdir "_preflight.py"
$probe | Out-File -FilePath $probeFile -Encoding utf8
& $Python $probeFile
$preflightOk = ($LASTEXITCODE -eq 0)

if (-not $preflightOk) {
    Write-Host ""
    Write-Host "PREFLIGHT FAILED. Nothing was launched." -ForegroundColor Red
    Write-Host "  Wrong interpreter is the usual cause. List envs:"
    Write-Host "      conda env list"
    Write-Host "  then re-run pointing at that env's python.exe:"
    Write-Host "      powershell -File scripts\run_jobB.ps1 -Python `"<path>\python.exe`""
    Write-Host "  If mirage or fm4ar fail inside the correct env, install them editable:"
    Write-Host "      cd `"$repo`";  <path>\python.exe -m pip install -e .  ;  <path>\python.exe -m pip install -e fm4ar"
    exit 1
}
Write-Host "Preflight OK."
Write-Host ""

# ---- the six arms. rad first: it is the one that settles the sigma column. ----
# h9_rad_sigma_s42 tests the 1.12-1.16 prediction against the published 1.3729,
# so it is the only arm whose result changes a decision. If one arm finishes,
# make it that one.
$arms = @(
    "h9_rad_sigma_s42",
    "h9_rad_sigma_s43",
    "h9_rad_sigma_s44",
    "h9_abc_sigma_s42",
    "h9_abc_sigma_s43",
    "h9_abc_sigma_s44"
)

$done = 0
$failed = @()
foreach ($arm in $arms) {
    $dir  = Join-Path $repo "configs\$arm"
    $best = Join-Path $dir "model__best.pt"
    $log  = Join-Path $logdir "$arm.log"

    if (Test-Path $best) {
        Write-Host "SKIP $arm, model__best.pt already present"; $done++
        continue
    }
    if (-not (Test-Path (Join-Path $dir "config.yaml"))) {
        Write-Host "SKIP $arm, no config.yaml"; $failed += $arm
        continue
    }

    Write-Host "=== $arm  start $(Get-Date -Format 'HH:mm:ss') ==="
    "=== $arm start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File -FilePath $log -Encoding utf8

    # 2>&1 merges stderr into stdout BEFORE the pipe, so a traceback lands in the
    # log as plain text instead of a PowerShell NativeCommandError wrapper.
    & $Python -u scripts\train.py --experiment-dir "configs\$arm" 2>&1 |
        Tee-Object -FilePath $log -Append

    if (Test-Path $best) {
        Write-Host "=== $arm  DONE $(Get-Date -Format 'HH:mm:ss') ==="; $done++
    } else {
        Write-Host "=== $arm  NO CHECKPOINT ===" -ForegroundColor Yellow
        Write-Host "    $log"
        Get-Content $log -Tail 15 | ForEach-Object { Write-Host "    $_" }
        $failed += $arm
        # An arm that dies for an environment reason will take the rest with it.
        # Stop rather than burn the queue, which is what happened at 16:50.
        if ($done -eq 0) {
            Write-Host ""
            Write-Host "First arm produced no checkpoint. Stopping instead of running the other five." -ForegroundColor Red
            break
        }
    }
}

Write-Host ""
Write-Host "Job B finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss').  $done of $($arms.Count) arms have checkpoints."
if ($failed.Count -gt 0) { Write-Host "  no checkpoint: $($failed -join ', ')" }
Write-Host "Health check with:"
Write-Host "  $Python scripts\check_run_health.py"
