param(
    [int]$RunnerProcessId,
    [int]$PollSeconds = 60
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$resultRoot = Join-Path $projectRoot "outputs\pytorch\final_experiment_sweep_1\validation_repeats_w30_qwmse_0p9_64_none"
$progressPath = Join-Path $resultRoot "progress.json"
$reportDirectory = Join-Path $projectRoot "outputs\pytorch\final_experiment_sweep_1"
$reportName = "cluster_comparison_rankings_w30_qwmse_0p9_rain_thresholds.tex"
$watchLog = Join-Path $resultRoot "pdf_watcher.log"

while ($true) {
    if (Test-Path -LiteralPath $progressPath) {
        $progress = Get-Content -LiteralPath $progressPath -Raw | ConvertFrom-Json
        if ([int]$progress.completed_runs -eq [int]$progress.total_runs) {
            break
        }
    }
    if (-not (Get-Process -Id $RunnerProcessId -ErrorAction SilentlyContinue)) {
        "Runner process $RunnerProcessId ended before all checkpoints completed." |
            Set-Content -LiteralPath $watchLog
        exit 1
    }
    Start-Sleep -Seconds $PollSeconds
}

Push-Location $reportDirectory
try {
    & pdflatex -interaction=nonstopmode -halt-on-error $reportName *>&1 |
        Set-Content -LiteralPath $watchLog
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    & pdflatex -interaction=nonstopmode -halt-on-error $reportName *>&1 |
        Add-Content -LiteralPath $watchLog
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
