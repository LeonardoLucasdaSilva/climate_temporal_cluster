# Repository Agent Guide

## Project Structure

This project studies Brazilian INMET daily climate data with temporal clustering
and cluster-specific LSTM precipitation models.

- `src/`: importable project code.
  - `config.py`: project paths such as `DATA_ROOT` and `OUTPUTS_DIR`.
  - `data/`: data loading, cleaning, and experiment output writers.
  - `methods/`: clustering pipelines, spectral clustering, sliding windows,
    sigma selection, and dimensionality reduction helpers.
  - `models/`: LSTM model code.
  - `evaluation/`: regression metrics, reports, and diagnostic plotting helpers.
- `experiments/`: runnable experiment scripts and experiment notes.
  - `lstm_cluster.py`: main LSTM-by-cluster sweep.
  - `temporary_experiments/`: older analysis scripts kept for review.
- `tests/`: unit tests for loaders and window feature creation.
- `data/`: local INMET data tree. Treat data files as local-only inputs.
- `outputs/`: generated experiment artifacts.
- Root scripts:
  - `run_experiments.py`: interactive launcher.
  - `verify_pipeline.py`: manual component verification script.

## Coding Conventions

- Prefer small, focused modules under `src/`; keep experiment scripts compact.
- Keep file-writing/report-generation helpers in `src/data/`, not embedded in
  experiment scripts.
- Use the current top-level imports from `src`, for example:
  - `from data.load_data import load_station_daily_data`
  - `from methods.cluster.cluster_pipeline import create_cluster_feature_matrix`
  - `from evaluation.metrics import calculate_regression_metrics`
- Do not reintroduce the old `src/climate_cluster/` package wrapper.
- Use type hints for public helpers and keep docstrings concise.
- Preserve existing behavior when refactoring. Pass experiment context
  explicitly instead of making shared modules depend on experiment globals.
- Avoid broad cleanup or formatting churn unrelated to the task.
- Keep generated outputs, data files, and local environment files out of code
  changes.

## Testing Procedure

Use the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

For quick syntax checks on touched files:

```powershell
.\.venv\Scripts\python.exe -m py_compile path\to\file.py
```

On this Windows setup, tests that use `tempfile.TemporaryDirectory()` may fail
inside a restricted sandbox with `PermissionError`. If that happens, rerun the
same test command outside the sandbox rather than changing test behavior.

## Do Not Modify Or Commit

Never modify or commit these unless the user explicitly asks:

- `.venv/`, `venv/`, `env/`, or any local Python environment.
- `.idea/`, `.vscode/`, editor metadata, or local IDE files.
- `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, and other caches.
- `data/inmet/**/*.csv` or other raw/local data files.
- `outputs/` experiment artifacts, generated reports, plots, metrics, and
  timestamped result folders.
- `.matplotlib_cache/` and `fontlist*.json`.
- Build/package artifacts such as `build/`, `dist/`, `*.egg-info/`, wheels, and
  coverage files.
- Large binary/generated files such as `*.pkl`, `*.pickle`, `*.npy`, `*.npz`,
  `*.html`, and `*.pdf`.

When these files appear in the working tree, treat them as local/generated
state. Do not delete or rewrite user data unless the user explicitly requested
that cleanup.
