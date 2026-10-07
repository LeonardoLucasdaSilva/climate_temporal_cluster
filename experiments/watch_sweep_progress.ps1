param(
    [Parameter(Mandatory = $true)]
    [string]$SweepDirectory,
    [Parameter(Mandatory = $true)]
    [int]$TotalConfigurations,
    [Parameter(Mandatory = $true)]
    [int]$ProcessId,
    [string]$OutputFileName = "sweep_progress.txt",
    [string]$TrainingSnapshot = "",
    [int]$RefreshSeconds = 15
)

$sweep = (Resolve-Path -LiteralPath $SweepDirectory).Path
$progressPath = Join-Path (Split-Path $sweep -Parent) $OutputFileName

do {
    $folders = @(Get-ChildItem -LiteralPath $sweep -Directory -ErrorAction SilentlyContinue)
    $started = [Math]::Min($folders.Count, $TotalConfigurations)
    $running = $null -ne (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)
    $finished = Test-Path -LiteralPath (Join-Path $sweep "sweep_results.csv")
    $completed = if ($finished) { $TotalConfigurations } elseif ($running) {
        [Math]::Max(0, $started - 1)
    } else {
        $started
    }
    $percent = if ($TotalConfigurations -gt 0) {
        [Math]::Floor(100 * $completed / $TotalConfigurations)
    } else { 0 }
    $filled = [Math]::Floor($percent / 2)
    $bar = ("#" * $filled).PadRight(50, "-")
    $current = if ($running -and $folders.Count -gt 0) {
        ($folders | Sort-Object LastWriteTime -Descending | Select-Object -First 1).Name
    } else { "none" }
    $status = if ($finished) { "complete" } elseif ($running) { "running" } else { "stopped" }
    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    $elapsed = if ($null -ne $process) {
        (Get-Date) - $process.StartTime
    } else { [TimeSpan]::Zero }
    $snapshotLines = if ($TrainingSnapshot -and (Test-Path -LiteralPath $TrainingSnapshot)) {
        @(Get-Content -LiteralPath $TrainingSnapshot)
    } else {
        @("Epoch details: unavailable; the active process writes epochs only to its console.")
    }

    $lines = @(
        "Status: $status"
        "Progress: [$bar] $percent%"
        "Completed configurations: $completed / $TotalConfigurations"
        "Configurations started: $started / $TotalConfigurations"
        "Current configuration: $current"
        "Process elapsed: $($elapsed.ToString('dd\.hh\:mm\:ss'))"
        "Updated: $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))"
        ""
        "Training details"
        "----------------"
        $snapshotLines
    )
    $lines | Set-Content -LiteralPath $progressPath -Encoding utf8

    if ($running -and -not $finished) {
        Start-Sleep -Seconds $RefreshSeconds
    }
} while ($running -and -not $finished)
