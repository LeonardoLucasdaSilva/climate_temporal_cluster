# Experiments

This folder contains experiment notes and older runnable scripts.

- `clustering_protocol.py`: shared experiment utilities for building window
  matrices, selecting sigma values, and dispatching clustering algorithms.
- `create_beamer_report.py`: command-line runner that creates a Beamer
  `beamer.tex` presentation and compiles `beamer.pdf` for one saved run under
  `outputs/`. It can list all available plots, select plots by relative path,
  glob, absolute path, or substring, and groups selected figures into analysis
  sections with a Madrid-style table-of-contents overview slide. The editable
  `PARAMS` list adds a first section with a ruled table of selected run
  parameters, including names copied from `run_experiment.py` constants.
- `create_meta_analysis_report.py`: editable and command-line runner that
  discovers saved LSTM and ARMA runs and writes a standalone `.tex`
  cross-experiment comparison. Its overall metric tables use MSE, MAE, and R2
  as rows, while lead-day tables use RMSE, MAE, and R2. Runs are globally
  numbered as columns, with a registry linking each number to the original
  experiment. Optional `START_DATE` and `END_DATE` values also create per-lead
  time-series plots beside the report, comparing observed precipitation against
  `Run 1`, `Run 2`, and every other selected run.
- `create_quantile_best_k_pdf.py`: directly generates one sectioned PDF table
  showing the strongest otherwise-identical K=1/higher-K pair for QWMSE(.9)
  and QWMSE(.85) at w=15 across the requested precipitation thresholds.
- `run_final_experiment_sweeps.py`: standalone, resumable, deterministic
  PyTorch launcher for `final_experiment_sweep_1` through `_5`. Each repetition
  sweeps windows, K, five automatic spectral sigmas, three quantile-loss
  configurations, three LSTM architectures (64, 128, and 64→32), MAE early
  stopping, and unstandardized precipitation. It writes only `metadata.json`
  and compressed predictions /
  histories per trained configuration, shares targets and cluster assignments,
  and still renders flat, per-horizon, and K-grouped ranking PDFs inside each
  repetition folder. Spectral graphs are reused across K values.
- `run_repeated_validation_rankings.py`: resumable validation-only rerunner for
  w=30, QWMSE(.9), and the 64/- architecture. It runs an unclustered K=1
  baseline plus K-means and five heuristic spectral sigmas for every K=2--9
  under seeds 42--46, without test prediction. It saves D+1 through D+5
  threshold MAE per seed, aggregates the mean and sample standard deviation,
  selects the lowest-mean spectral sigma per K/table, and replaces the matching
  ranking TeX.
- `watch_repeated_validation_pdf.ps1`: optional hidden completion watcher used
  to compile that repeated-validation TeX after every checkpoint finishes.
- `random_search/`: editable, resumable two-stage PyTorch hyperparameter
  search. It screens random configurations over K=1,3,5,7,9 and K-means plus
  heuristic-sigma spectral clustering, retains the top validation-MAE decile,
  then reruns exact finalists under new seeds and ranks mean validation MAE
  with its sample standard deviation. Confirmation folders include
  train/validation curves, and the test split is never evaluated.
- `reconstruct_final_experiment_artifacts.py`: recreates the normal CSVs,
  plots, TeX, and PDF for a selected compact final configuration without
  retraining it.
- `temporary_experiments/`: older experiment scripts kept temporarily so they
  can be reviewed, saved elsewhere, or folded back into the organized package.

Run the main experiment from the project root:

```powershell
lstm-cluster
```

Run or resume the compact final sweeps, repair their rankings, or reconstruct
one selected configuration:

```powershell
.\.venv\Scripts\python.exe -u experiments\run_final_experiment_sweeps.py
.\.venv\Scripts\python.exe experiments\run_final_experiment_sweeps.py --render-only --repetitions 1
.\.venv\Scripts\python.exe experiments\reconstruct_final_experiment_artifacts.py outputs\pytorch\final_experiment_sweep_1\compact_results\<batch>\<configuration>
.\.venv\Scripts\python.exe -u experiments\run_repeated_validation_rankings.py
```

Run or resume the two-stage validation-only random search with:

```powershell
.\.venv\Scripts\python.exe -u experiments\random_search\run_random_search.py
```

Its outputs are isolated under `outputs/pytorch/random_search/<search_id>/`.
See `random_search/random_search.md` for the full framework and output layout.

Benchmark alternative training batch sizes against one completed final-sweep
configuration without modifying the official sweep artifacts:

```powershell
.\.venv\Scripts\python.exe -u experiments\benchmark_final_sweep_batch_sizes.py
```

The benchmark reuses the saved data split and cluster assignments, then records
runtime, GPU utilization, epoch counts, and validation/test metrics in
`outputs/pytorch/batch_size_benchmark.csv`.

The active final sweep uses learning rate `1e-3` and batch size 8. A
representative batch-size benchmark
found it about 1.30 times faster than batch size 4 while preserving comparable
validation and test metrics; larger tested batches were faster but materially
degraded test accuracy. Cluster-specific LSTMs use at most two training workers;
the two-worker benchmark was 1.74 times faster than sequential training and
reproduced all saved prediction and history arrays exactly.

While the final sweep runs, `SWEEP_PROGRESS.md` in the project root is updated
after every configuration checkpoint. It reports trained and invalid/skipped
counts separately, plus the total processed and remaining configurations. The
per-repetition machine-readable counterpart remains at
`outputs/pytorch/final_experiment_sweep_<N>/progress.txt`.

Benchmark bounded two-worker cluster-model training against a completed
sequential configuration with:

```powershell
.\.venv\Scripts\python.exe -u experiments\benchmark_final_sweep_parallel_training.py
```

Run an isolated paired pilot of a 32-unit second LSTM layer on matched K=1,
K=3, and K=5 configurations with:

```powershell
.\.venv\Scripts\python.exe -u experiments\benchmark_second_lstm_layer.py
```

Create a presentation from one saved configuration run by editing the block at
the top of `create_beamer_report.py`:

```python
RUN_DIR = PROJECT_ROOT / "outputs" / "lstm_cluster_sweep_RS_A801_YYYYMMDD_HHMMSS" / "RS_A801_w15_k03_kmeans"
SELECTED_PLOTS = [
    "prediction_overview/02_predictions_vs_actual.png",
    "cluster_prediction_scatter/*.png",
    "residual_diagnostics/*.png",
]
PARAMS = ["LEARNING_RATE", "EPOCHS", "LSTM_UNITS_1", "CLUSTERING_ALGORITHM"]
```

Create a metric meta-analysis by editing the block at the top of
`create_meta_analysis_report.py`:

```python
EXPERIMENT_PATHS = [
    PROJECT_ROOT / "outputs" / "dd_mm_yy" / "lstm_cluster_sweep_...",
    PROJECT_ROOT / "outputs" / "dd_mm_yy" / "ARMA" / "arma_sweep_...",
]
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "meta_analysis.tex"
RUNS_PER_TABLE = 10
START_DATE = "2018-10-01"
END_DATE = "2018-12-31"
```

Alternatively, pass all inputs on the command line:

```powershell
python experiments\create_meta_analysis_report.py <experiment_1> <experiment_2> --output outputs\meta_analysis.tex --start-date 2018-10-01 --end-date 2018-12-31
```

The runner returns a concise error for invalid inputs. Raw artifacts and
common-date-aligned `comparative_metrics.csv` inputs belong in separate
reports; multiple aligned sources must describe the same lead days, date
intervals, and common test sample counts.
