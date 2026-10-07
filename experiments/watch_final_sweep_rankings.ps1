param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectDirectory,
    [Parameter(Mandatory = $true)]
    [int]$ExperimentProcessId,
    [int]$RefreshSeconds = 300
)

$project = (Resolve-Path -LiteralPath $ProjectDirectory).Path
$python = Join-Path $project ".venv\Scripts\python.exe"
$renderer = Join-Path $project "experiments\run_final_experiment_sweeps.py"
$outputRoot = Join-Path $project "outputs\pytorch"

function Update-RepetitionReport([int]$repetition, [switch]$Force) {
    $repetitionDirectory = Join-Path $outputRoot "final_experiment_sweep_$repetition"
    if (-not (Test-Path -LiteralPath $repetitionDirectory)) {
        return
    }
    $manifest = Join-Path $repetitionDirectory "configuration_manifest.csv"
    $completedConfigurations = if (Test-Path -LiteralPath $manifest) {
        @(Import-Csv -LiteralPath $manifest).Count
    } else {
        0
    }
    $completedBatches = [math]::Floor($completedConfigurations / 225)
    $marker = Join-Path $repetitionDirectory ".rendered_batch_count"
    $renderedBatches = if (Test-Path -LiteralPath $marker) {
        [int](Get-Content -LiteralPath $marker -Raw)
    } else {
        -1
    }
    if ($Force -or $completedBatches -gt $renderedBatches) {
        & $python $renderer --render-only --repetitions $repetition
        if ($LASTEXITCODE -eq 0) {
            Set-Content -LiteralPath $marker -Value $completedBatches -Encoding ascii
        }
    }
}

do {
    $running = $null -ne (
        Get-Process -Id $ExperimentProcessId -ErrorAction SilentlyContinue
    )
    foreach ($repetition in 1..5) {
        Update-RepetitionReport -repetition $repetition
    }
    if ($running) {
        Start-Sleep -Seconds $RefreshSeconds
    }
} while ($running)

foreach ($repetition in 1..5) {
    Update-RepetitionReport -repetition $repetition -Force
}
