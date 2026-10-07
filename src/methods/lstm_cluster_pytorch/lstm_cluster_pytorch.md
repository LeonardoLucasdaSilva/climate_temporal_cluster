# Local PyTorch LSTM-cluster pipeline

This git-ignored implementation mirrors `methods.lstm_cluster.pipeline` but
uses a PyTorch LSTM. Its independent runner is `run_experiment.py`, and its
artifacts are written beneath `outputs/pytorch/<date>/`.

Install PyTorch in the environment you intend to use. For a CUDA build, use the
install command generated for Windows, Pip, and your supported CUDA version at
<https://pytorch.org/get-started/locally/>. Then verify CUDA and run:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
.\.venv\Scripts\python.exe -u src\methods\lstm_cluster_pytorch\run_experiment.py
```

For an existing project environment that already has the scientific packages,
only PyTorch needs to be added. The included `requirements-pytorch.txt` is also
available when creating a separate environment from scratch; it intentionally
does not install TensorFlow.

The runner defaults to `REQUIRE_GPU = True`, so it fails early rather than
silently training on the CPU. Set it to `False` only when CPU fallback is
intentional.

The PyTorch pipeline also exposes a strict `validation_only` execution path.
It predicts training and validation rows but does not predict, evaluate, or
rank the test split. `save_validation_history=True` adds per-cluster four-panel
train/validation curves without enabling test output. The two-stage framework
in `experiments/random_search/` uses this path for both screening and
multi-seed confirmation.

`CREATE_REPORT = True` by default, so every completed configuration compiles
its own `experiment_report.pdf` beneath the PyTorch output folder.

After a sweep finishes successfully, the pipeline removes empty directories
beneath that sweep's output folder. This clears configuration folders for
invalid cluster distributions that were skipped before artifacts were written.

At startup and periodically during completed-configuration ranking updates,
the pipeline additively discovers local result folders missing from the
backend registry. Saved trained D+1 through D+H metrics are stored in
`outputs/pytorch/experiment_horizon_rankings.csv`; every day is rendered to its
own `experiment_rankings_horizon_dNN.{csv,tex,pdf}` artifact set, including
independently ranked above-0/10/20/30 mm and daily 95th/99th-percentile tables.
Rows from the other experiment computer remain in the registry
even when their result folders are not present locally.
Ranking CSVs retain both recurrent-layer sizes, and rendered tables show
`U_2=-` for single-layer models.

Multi-algorithm sweeps create one unclustered `K=1` baseline per window size;
the baseline is not repeated for each clustering algorithm or spectral sigma.

The reproducibility launcher `experiments/run_final_experiment_sweeps.py`
defines the five `final_experiment_sweep_N` runs. It validates 675 model
configurations per repetition, uses MAE-only early stopping without
precipitation standardization, and is deliberately separate from this normal
pipeline. It uses deterministic PyTorch execution with seed 42, reuses each
spectral graph across K, resumes compact results, and stores only metadata plus
compressed predictions/history per configuration. It writes overall and D+1
through D+5 rankings inside each repetition folder and renders matched
comparison tables. Each table fixes loss, patience, U1, U2, precipitation
standardization, and window size while clustering method and K vary by row.
For each K, a table retains only the spectral sigma with its lowest MAE on that
table's own precipitation subset. Non-spectral comparison rows remain.
The rendered tables omit the patience and R2 columns; patience is still used
internally when matching otherwise identical configurations.

For each D+1 through D+5 horizon, report rendering also creates separate
cluster-comparison PDFs for actual precipitation above 0 mm, above 10 mm, and
above 30 mm. Metrics in these dedicated files use only the corresponding rain
subset, while the overall reports remain available.

Run all repetitions with:

```powershell
.\.venv\Scripts\python.exe -u experiments\run_final_experiment_sweeps.py
```

Use `--dry-run`, `--repetitions 1 2`, or `--render-only` for validation,
partial execution, or report regeneration.
Use `experiments/reconstruct_final_experiment_artifacts.py` with a compact
configuration folder to recreate its ordinary plots and PDF without retraining.
The runner selects Matplotlib's non-interactive `Agg` backend because experiment
plots are written to files and should not require a desktop Tcl/Tk runtime.
The final sweep bounds cluster-specific GPU training at two worker processes;
`K=1` remains sequential. A saved-result benchmark must show exact prediction
and history equality before this parallel path is enabled.
