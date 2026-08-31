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

`CREATE_REPORT = True` by default, so every completed configuration compiles
its own `experiment_report.pdf` beneath the PyTorch output folder.
