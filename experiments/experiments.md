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
- `temporary_experiments/`: older experiment scripts kept temporarily so they
  can be reviewed, saved elsewhere, or folded back into the organized package.

Run the main experiment from the project root:

```powershell
lstm-cluster
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
