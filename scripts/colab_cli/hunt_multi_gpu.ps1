param(
    [string]$Session = "exp-rl",
    [int]$MaxRounds = 30
)
$ErrorActionPreference = "Continue"
$rootDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$c = Join-Path $rootDir ".venv\Scripts\colab.exe"
$logFile = Join-Path $PSScriptRoot ("hunt_" + $Session + ".log")

function Log-Msg {
    param([string]$Msg)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $Msg"
    Write-Host $line
    Add-Content -Path $logFile -Value $line -Encoding UTF8
}

function Invoke-ColabWithTimeout {
    param([string[]]$ArgList, [int]$TimeoutSec = 240, [string]$Retries = "2", [string]$Backoff = "6")
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $c
    $psi.Arguments = ($ArgList | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' '
    $psi.EnvironmentVariables["COLAB_CLI_ASSIGN_RETRIES"] = $Retries
    $psi.EnvironmentVariables["COLAB_CLI_RETRY_BACKOFF"] = $Backoff
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $p = New-Object System.Diagnostics.Process
    $p.StartInfo = $psi
    [void]$p.Start()
    $outTask = $p.StandardOutput.ReadToEndAsync()
    $errTask = $p.StandardError.ReadToEndAsync()
    if (-not $p.WaitForExit($TimeoutSec * 1000)) {
        try { $p.Kill() } catch {}
        return @{ Code = -1; Output = "TIMEOUT after ${TimeoutSec}s" }
    }
    $combined = ($outTask.Result + "`n" + $errTask.Result).Trim()
    return @{ Code = $p.ExitCode; Output = $combined }
}

$pyInfo = @'
import os, subprocess
gpu = subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total,compute_cap","--format=csv,noheader"], text=True).strip()
ram = int([l for l in open("/proc/meminfo") if l.startswith("MemTotal:")][0].split()[1]) / 1024 / 1024
print(f"GPU: {gpu} | CPU: {os.cpu_count()} | RAM: {ram:.1f} GB")
'@
$tmpPy = Join-Path $PSScriptRoot ("_info_" + $Session + ".py")
Set-Content -Path $tmpPy -Value $pyInfo -Encoding ASCII

# Hunt across all top-tier GPUs: G4 (96GB Blackwell, best price/perf) -> H100 (80GB) -> A100 (40GB)
$targets = @(
    @{ Name = "G4";   Args = @("new", "-s", $Session, "--gpu", "G4");   Retries = "2"; Timeout = 190 },
    @{ Name = "H100"; Args = @("new", "-s", $Session, "--gpu", "H100"); Retries = "1"; Timeout = 140 },
    @{ Name = "A100"; Args = @("new", "-s", $Session, "--gpu", "A100"); Retries = "2"; Timeout = 190 }
)

Log-Msg "=== Starting fast GPU hunter for session '$Session' (G4 96GB -> H100 80GB -> A100 40GB) ==="

for ($r = 1; $r -le $MaxRounds; $r++) {
    Log-Msg "--- Round #$r ---"
    foreach ($t in $targets) {
        Log-Msg "Trying $($t.Name) (retries=$($t.Retries))..."
        $res = Invoke-ColabWithTimeout -ArgList $t.Args -TimeoutSec $t.Timeout -Retries $t.Retries -Backoff "6"
        if ($res.Code -eq 0) {
            Log-Msg "ACQUIRED $Session=$($t.Name) on round #$r!"
            $gRes = Invoke-ColabWithTimeout -ArgList @("exec", "-s", $Session, "--timeout", "45", "-f", $tmpPy) -TimeoutSec 60
            Log-Msg "HW_INFO: $($gRes.Output)"
            Remove-Item $tmpPy -Force -ErrorAction SilentlyContinue
            exit 0
        }
        $errLine = ($res.Output -split "`r?`n" | Where-Object { $_ -match "ColabRequestError|Service Unavailable|TooManyAssignmentsError|TIMEOUT|ConnectionResetError" } | Select-Object -Last 1)
        if (-not $errLine) { $errLine = "exit=$($res.Code)" }
        Log-Msg "Failed $($t.Name): $errLine"
        if ($res.Output -match "(?m)^TooManyAssignmentsError:") {
            Log-Msg "Hit TooManyAssignmentsError; stopping hunter."
            Remove-Item $tmpPy -Force -ErrorAction SilentlyContinue
            exit 2
        }
        $sChk = Invoke-ColabWithTimeout -ArgList @("sessions") -TimeoutSec 30
        if ($sChk.Output -match [regex]::Escape($Session)) {
            Log-Msg "Late-assigned session '$Session' detected!"
            $gRes = Invoke-ColabWithTimeout -ArgList @("exec", "-s", $Session, "--timeout", "45", "-f", $tmpPy) -TimeoutSec 60
            Log-Msg "HW_INFO: $($gRes.Output)"
            Remove-Item $tmpPy -Force -ErrorAction SilentlyContinue
            exit 0
        }
        Start-Sleep -Seconds 4
    }
    Log-Msg "Round #$r finished; sleeping 60s before next round..."
    Start-Sleep -Seconds 60
}
Remove-Item $tmpPy -Force -ErrorAction SilentlyContinue
Log-Msg "Reached max rounds without acquiring GPU."
exit 1
