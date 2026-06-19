# Experiments

This folder contains runnable experiment scripts.

- `lstm_cluster.py`: the main LSTM-by-cluster precipitation experiment.
- `clustering_protocol.py`: shared experiment utilities for building window
  matrices, selecting sigma values, and dispatching clustering algorithms.
- `temporary_experiments/`: older experiment scripts kept temporarily so they
  can be reviewed, saved elsewhere, or folded back into the organized package.

Run the main experiment from the project root:

```powershell
.\.venv\Scripts\python.exe experiments\lstm_cluster.py
```

