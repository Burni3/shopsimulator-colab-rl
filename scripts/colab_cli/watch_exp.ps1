param(
    [Parameter(Mandatory=$true)][string]$Session,
    [int]$PollSec = 180,
    [int]$MaxPolls = 120,
    [switch]$NoStop
)
$ErrorActionPreference = "Continue"
$rootDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$c = Join-Path $rootDir ".venv\Scripts\colab.exe"
$checkPy = Join-Path $PSScriptRoot "check_exp.py"
$outDir = Join-Path $PSScriptRoot ("results_" + $Session)
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

# Keep Windows awake while monitoring Colab WebSocket keep-alive
try {
    if (-not ("ColabSleepGuard" -as [type])) {
        Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class ColabSleepGuard {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
    }
    # ES_CONTINUOUS (0x80000000) | ES_SYSTEM_REQUIRED (0x00000001) | ES_DISPLAY_REQUIRED (0x00000002)
    # Prevents Windows 11 Modern Standby from dropping Wi-Fi/Colab WebSocket!
    [void][ColabSleepGuard]::SetThreadExecutionState([uint32]2147483651)
    Write-Host "[$Session] Windows Modern-Standby awake-lock enabled (system + display required)."
} catch {
    Write-Host "[$Session] WARN: Could not enable Windows awake-lock: $_"
}

function Invoke-ColabWithTimeout {
    param([string[]]$ArgList, [int]$TimeoutSec = 90)
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $c
    $psi.Arguments = ($ArgList | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' '
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

for ($i = 1; $i -le $MaxPolls; $i++) {
    $ts = Get-Date -Format "HH:mm:ss"
    Write-Host "=== [$Session][$ts] Poll #$i ==="
    $sRes = Invoke-ColabWithTimeout -ArgList @("sessions") -TimeoutSec 45
    if ($sRes.Code -ne 0 -or $sRes.Output -notmatch [regex]::Escape($Session)) {
        Write-Host "WARN/ERR sessions (code=$($sRes.Code)): $($sRes.Output)"
        if ($sRes.Code -eq 0 -and $sRes.Output -match "No active sessions") {
            Write-Host "ERROR: $Session session disappeared!"
            exit 2
        }
        Start-Sleep -Seconds 20
        continue
    }

    $eRes = Invoke-ColabWithTimeout -ArgList @("exec", "-s", $Session, "--timeout", "60", "-f", $checkPy) -TimeoutSec 90
    if ($eRes.Code -ne 0 -or $eRes.Output -match "Connection was lost|TIMEOUT") {
        Write-Host "Transient exec hiccup; retrying once in 5s..."
        Start-Sleep -Seconds 5
        $eRes = Invoke-ColabWithTimeout -ArgList @("exec", "-s", $Session, "--timeout", "60", "-f", $checkPy) -TimeoutSec 90
    }
    Write-Host $eRes.Output

    if ($eRes.Output -match "rc=0\b") {
        Write-Host "=== [$Session] ALL QUEUED EXPERIMENTS SUCCEEDED (rc=0)! Downloading logs... ==="
        & $c download -s $Session /content/exp_runner.log (Join-Path $outDir "exp_runner.log")
        if ($NoStop) {
            Write-Host "SESSION_DONE_KEPT_ALIVE=$Session (NoStop enabled; keeping GPU alive for next run)"
            exit 0
        }
        # Wait 30s for DriveFS flush before stopping
        Start-Sleep -Seconds 30
        & $c stop -s $Session
        & $c usage
        Write-Host "SESSION_DONE_AND_STOPPED=$Session"
        exit 0
    }
    if ($eRes.Output -match "rc=([1-9]\d*)\b") {
        Write-Host "[$Session] FAILED ($($Matches[0])); downloading log and keeping session alive for inspection."
        & $c download -s $Session /content/exp_runner.log (Join-Path $outDir "exp_runner.log")
        exit 1
    }

    Start-Sleep -Seconds $PollSec
}
Write-Host "[$Session] Reached max poll iterations."
exit 3
