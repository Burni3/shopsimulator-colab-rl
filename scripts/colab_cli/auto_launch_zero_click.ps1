param(
    [Parameter(Mandatory=$true)][string]$Session,
    [Parameter(Mandatory=$true)][string]$QueueFile,
    [int]$MaxHuntRounds = 25
)
$ErrorActionPreference = "Continue"
$rootDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$c = Join-Path $rootDir ".venv\Scripts\colab.exe"
$hfTok = Join-Path $rootDir "request-demo\.hf_token"

Write-Host "=== [$Session] Step 1: Hunt largest available GPU (H100 -> G4 -> A100) ==="
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "hunt_multi_gpu.ps1") -Session $Session -MaxRounds $MaxHuntRounds
if ($LASTEXITCODE -ne 0) {
    Write-Host "[$Session] Failed to acquire GPU (exit=$LASTEXITCODE)"
    exit $LASTEXITCODE
}

Write-Host "=== [$Session] Step 2: Zero-click upload scripts & HF token ==="
& $c upload -s $Session $hfTok "/content/.hf_token"
& $c upload -s $Session (Join-Path $PSScriptRoot "restore_from_hf.sh") "/content/restore_from_hf.sh"
& $c upload -s $Session (Join-Path $PSScriptRoot "run_experiment.sh") "/content/run_experiment.sh"
& $c upload -s $Session (Join-Path $PSScriptRoot "shop_innovations.py") "/content/shop_innovations.py"
& $c upload -s $Session (Join-Path $PSScriptRoot $QueueFile) "/content/exp_queue.txt"

Write-Host "=== [$Session] Step 3: Launch detached zero-click experiment queue ==="
& $c exec -s $Session --timeout 60 -f (Join-Path $PSScriptRoot "launch_exp.py")
if ($LASTEXITCODE -ne 0) {
    Write-Host "[$Session] Failed to launch experiment queue!"
    exit $LASTEXITCODE
}

Write-Host "=== [$Session] Step 4: Start automated watcher & auto-stop ==="
Start-Sleep -Seconds 30
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "watch_exp.ps1") -Session $Session -PollSec 120 -MaxPolls 200
exit $LASTEXITCODE
