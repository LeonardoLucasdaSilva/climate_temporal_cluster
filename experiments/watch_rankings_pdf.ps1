param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectDirectory,
    [Parameter(Mandatory = $true)]
    [int]$ExperimentProcessId,
    [int]$RefreshSeconds = 30
)

$project = (Resolve-Path -LiteralPath $ProjectDirectory).Path
$tex = Join-Path $project "outputs\pytorch\experiment_rankings.tex"
$pdf = Join-Path $project "outputs\pytorch\experiment_rankings.pdf"
$python = Join-Path $project ".venv\Scripts\python.exe"

do {
    $running = $null -ne (
        Get-Process -Id $ExperimentProcessId -ErrorAction SilentlyContinue
    )
    $needsCompile = (Test-Path -LiteralPath $tex) -and (
        -not (Test-Path -LiteralPath $pdf) -or
        (Get-Item -LiteralPath $tex).LastWriteTime -gt
            (Get-Item -LiteralPath $pdf).LastWriteTime
    )
    if ($needsCompile) {
        & $python -c (
            "from pathlib import Path; " +
            "from methods.lstm_cluster.report import compile_tex; " +
            "compile_tex(Path(r'$tex'))"
        )
    }
    if ($running) {
        Start-Sleep -Seconds $RefreshSeconds
    }
} while ($running)

# Catch a ranking update written immediately before the process exited.
if ((Test-Path -LiteralPath $tex) -and (
    -not (Test-Path -LiteralPath $pdf) -or
    (Get-Item -LiteralPath $tex).LastWriteTime -gt
        (Get-Item -LiteralPath $pdf).LastWriteTime
)) {
    & $python -c (
        "from pathlib import Path; " +
        "from methods.lstm_cluster.report import compile_tex; " +
        "compile_tex(Path(r'$tex'))"
    )
}
