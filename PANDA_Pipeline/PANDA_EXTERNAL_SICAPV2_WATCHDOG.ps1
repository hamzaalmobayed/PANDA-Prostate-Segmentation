param([Parameter(Mandatory=$true)][string]$PipelineDir)
$ErrorActionPreference = "Continue"
$pollSeconds = 10
$scriptName = "run_external_sicapv2_test.py"
function Test-Ready([string]$Dir) {
    try {
        if (-not (Test-Path -LiteralPath $Dir)) { return $false }
        $script = Join-Path $Dir $scriptName
        if (-not (Test-Path -LiteralPath $script)) { return $false }
        Get-ChildItem -LiteralPath $Dir -ErrorAction Stop | Select-Object -First 1 | Out-Null
        return $true
    } catch { return $false }
}
Write-Host "[SICAPv2 WATCHDOG] Running from internal Windows storage."
Write-Host "[SICAPv2 WATCHDOG] Pipeline: $PipelineDir"
while ($true) {
    while (-not (Test-Ready $PipelineDir)) {
        Write-Host "[SICAPv2 WATCHDOG] Package/SSD unavailable. Waiting $pollSeconds sec..."
        Start-Sleep -Seconds $pollSeconds
    }
    try {
        Push-Location -LiteralPath $PipelineDir
        $env:NO_ALBUMENTATIONS_UPDATE = "1"
        & python $scriptName
        $code = $LASTEXITCODE
        Pop-Location
        if ($code -eq 0) {
            Write-Host "[SICAPv2 WATCHDOG] External test completed successfully."
            exit 0
        }
        Write-Host "[SICAPv2 WATCHDOG] Python exited with code $code. Retrying in $pollSeconds sec..."
    } catch {
        try { Pop-Location } catch {}
        Write-Host "[SICAPv2 WATCHDOG] $($_.Exception.Message)"
    }
    Start-Sleep -Seconds $pollSeconds
}
