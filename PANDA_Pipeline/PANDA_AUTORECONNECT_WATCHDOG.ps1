param(
    [Parameter(Mandatory=$true)]
    [string]$PipelineDir
)

$ErrorActionPreference = "Continue"
$pollSeconds = 10

function Test-PipelineReady {
    param([string]$Dir)
    try {
        if (-not (Test-Path -LiteralPath $Dir)) { return $false }
        $runner = Join-Path $Dir "run_full_panda_paper.py"
        if (-not (Test-Path -LiteralPath $runner)) { return $false }

        # Force a real directory read so a half-mounted/disconnected drive is not
        # mistaken for healthy storage.
        Get-ChildItem -LiteralPath $Dir -ErrorAction Stop | Select-Object -First 1 | Out-Null
        return $true
    } catch {
        return $false
    }
}

Write-Host "==============================================================="
Write-Host "[WATCHDOG] PANDA auto-reconnect watchdog is running from C:."
Write-Host "[WATCHDOG] Pipeline: $PipelineDir"
Write-Host "[WATCHDOG] Do NOT close this window."
Write-Host "==============================================================="

while ($true) {
    while (-not (Test-PipelineReady -Dir $PipelineDir)) {
        Write-Host "[WATCHDOG] SSD/pipeline unavailable. Waiting $pollSeconds seconds..."
        Start-Sleep -Seconds $pollSeconds
    }

    Write-Host "[WATCHDOG] Storage is readable. Starting/resuming pipeline automatically..."

    try {
        Push-Location -LiteralPath $PipelineDir
        $env:NO_ALBUMENTATIONS_UPDATE = "1"

        # Run synchronously. The Python code itself already waits/retries on
        # recognized SSD interruptions. If Python exits anyway, this watchdog
        # remains alive on the internal system drive and restarts it.
        & python "run_full_panda_paper.py"
        $exitCode = $LASTEXITCODE

        Pop-Location
        Write-Host "[WATCHDOG] Python exited with code $exitCode."
    } catch {
        try { Pop-Location } catch {}
        Write-Host "[WATCHDOG] Launch/runtime exception: $($_.Exception.Message)"
    }

    Write-Host "[WATCHDOG] Waiting $pollSeconds seconds before automatic recovery..."
    Start-Sleep -Seconds $pollSeconds
}
