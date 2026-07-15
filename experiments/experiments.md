# Experiments

This folder contains experiment notes and older runnable scripts.

- `clustering_protocol.py`: shared experiment utilities for building window
  matrices, selecting sigma values, and dispatching clustering algorithms.
- `create_beamer_report.py`: command-line runner that creates a Beamer
  `beamer.tex` presentation and compiles `beamer.pdf` for one saved run under
  `outputs/`. It can list all available plots, select plots by relative path,
  glob, absolute path, or substring, and groups selected figures into analysis
  sections with a clickable overview slide.
- `temporary_experiments/`: older experiment scripts kept temporarily so they
  can be reviewed, saved elsewhere, or folded back into the organized package.

Run the main experiment from the project root:

```powershell
lstm-cluster
```

Create a presentation from one saved configuration run by editing the block at
the top of `create_beamer_report.py`:

```python
RUN_DIR = PROJECT_ROOT / "outputs" / "lstm_cluster_sweep_RS_A801_YYYYMMDD_HHMMSS" / "RS_A801_w15_k03_kmeans_sigma_na"
SELECTED_PLOTS = [
    "prediction_overview/02_predictions_vs_actual.png",
    "cluster_prediction_scatter/*.png",
    "residual_diagnostics/*.png",
]
```
