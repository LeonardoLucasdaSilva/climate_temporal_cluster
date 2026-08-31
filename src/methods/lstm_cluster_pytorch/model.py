"""PyTorch implementation of the cluster-specific precipitation LSTM."""

from __future__ import annotations

from copy import deepcopy
from typing import Sequence

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # Give the runner a useful error instead of a vague import failure.
    raise ImportError(
        "The PyTorch pipeline requires PyTorch. Install it in the active "
        "environment using the command documented in lstm_cluster_pytorch.md."
    ) from exc


SUPPORTED_LOSS_FUNCTIONS = (
    "mean_squared_error",
    "mse",
    "mean_absolute_error",
    "mae",
    "huber",
    "weighted_mse_loss",
    "quantile_weighted_mse",
)
SUPPORTED_EARLY_STOPPING_METRICS = ("loss", "mse", "mae", "r2")


class _TorchLSTMNetwork(nn.Module):
    """LSTM architecture matching the TensorFlow experiment model."""

    def __init__(
        self,
        input_features: int,
        lstm_units: int,
        lstm_units_2: int | None,
        dropout_rate: float,
        output_units: int,
    ) -> None:
        super().__init__()
        self.first_lstm = nn.LSTM(input_features, lstm_units, batch_first=True)
        self.second_lstm = (
            nn.LSTM(lstm_units, lstm_units_2, batch_first=True)
            if lstm_units_2 is not None
            else None
        )
        final_lstm_units = lstm_units_2 if lstm_units_2 is not None else lstm_units
        self.dropout = nn.Dropout(dropout_rate)
        self.dense_1 = nn.Linear(final_lstm_units, 16)
        self.dense_2 = nn.Linear(16, 8)
        self.output = nn.Linear(8, output_units)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence, (hidden, _cell) = self.first_lstm(inputs)
        if self.second_lstm is not None:
            _sequence, (hidden, _cell) = self.second_lstm(sequence)
        features = self.dropout(hidden[-1])
        features = self.dropout(torch.relu(self.dense_1(features)))
        features = torch.relu(self.dense_2(features))
        return self.output(features)


class LSTMPrecipitationPredictor:
    """Train and evaluate the pipeline's precipitation model with PyTorch."""

    def __init__(
        self,
        input_shape: tuple[int, ...],
        lstm_units: int = 64,
        lstm_units_2: int | None = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        random_state: int = 42,
        loss_function: str = "mean_squared_error",
        loss_quantile_thresholds_mm: Sequence[float] | None = None,
        loss_quantile_weights: Sequence[float] | None = None,
        output_units: int = 1,
        loss_alpha: float | None = None,
        require_gpu: bool = False,
        gpu_index: int = 0,
    ) -> None:
        if len(input_shape) != 2:
            raise ValueError("input_shape must be (sequence_length, n_features).")
        if output_units <= 0:
            raise ValueError("output_units must be positive.")
        if not np.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be a finite non-negative value.")
        normalized_loss = str(loss_function).strip().lower()
        if normalized_loss not in SUPPORTED_LOSS_FUNCTIONS:
            supported = ", ".join(SUPPORTED_LOSS_FUNCTIONS)
            raise ValueError(
                f"Unsupported loss_function: {loss_function!r}. Use one of: {supported}"
            )
        if gpu_index < 0:
            raise ValueError("gpu_index must be a non-negative integer.")
        if require_gpu and not torch.cuda.is_available():
            raise RuntimeError(
                "PyTorch did not detect a CUDA GPU. Install a CUDA-enabled "
                "PyTorch build and a compatible NVIDIA driver."
            )
        if torch.cuda.is_available() and gpu_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested GPU index {gpu_index}, but PyTorch detected only "
                f"{torch.cuda.device_count()} CUDA GPU(s)."
            )

        self.input_shape = tuple(int(value) for value in input_shape)
        self.output_units = int(output_units)
        self.random_state = int(random_state)
        self.loss_function = normalized_loss
        self.loss_alpha = loss_alpha
        self.device = torch.device(
            f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu"
        )
        self.device_name = str(self.device)
        self.history: dict[str, list[float]] | None = None

        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        self.model = _TorchLSTMNetwork(
            input_features=self.input_shape[-1],
            lstm_units=int(lstm_units),
            lstm_units_2=(None if lstm_units_2 is None else int(lstm_units_2)),
            dropout_rate=float(dropout_rate),
            output_units=self.output_units,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        self._thresholds = self._validated_thresholds(loss_quantile_thresholds_mm)
        self._quantile_weights = self._validated_quantile_weights(
            loss_quantile_weights
        )
        self._validate_loss_configuration()

    @staticmethod
    def _validated_thresholds(values: Sequence[float] | None) -> torch.Tensor | None:
        if values is None:
            return None
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 1 or (len(array) and np.any(np.diff(array) <= 0)):
            raise ValueError("loss quantile thresholds must be one-dimensional and increasing.")
        return torch.as_tensor(array)

    @staticmethod
    def _validated_quantile_weights(
        values: Sequence[float] | None,
    ) -> torch.Tensor | None:
        if values is None:
            return None
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 1 or np.any(array <= 0):
            raise ValueError("loss quantile weights must be one-dimensional and positive.")
        return torch.as_tensor(array)

    def _validate_loss_configuration(self) -> None:
        if self.loss_function == "weighted_mse_loss":
            if (
                isinstance(self.loss_alpha, (bool, np.bool_))
                or self.loss_alpha is None
                or not np.isfinite(float(self.loss_alpha))
                or float(self.loss_alpha) <= 0
            ):
                raise ValueError("weighted_mse_loss requires a finite positive alpha.")
        if self.loss_function == "quantile_weighted_mse":
            if self._thresholds is None or self._quantile_weights is None:
                raise ValueError(
                    "quantile_weighted_mse requires thresholds and weights."
                )
            if len(self._quantile_weights) != len(self._thresholds) + 1:
                raise ValueError(
                    "weights must contain exactly len(thresholds) + 1 values."
                )

    def _loss(self, targets: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
        if self.loss_function in {"mean_absolute_error", "mae"}:
            return torch.mean(torch.abs(targets - predictions))
        if self.loss_function == "huber":
            return nn.functional.huber_loss(predictions, targets, delta=1.0)
        squared_error = torch.square(targets - predictions)
        if self.loss_function == "weighted_mse_loss":
            squared_error = (1.0 + float(self.loss_alpha) * targets) * squared_error
        elif self.loss_function == "quantile_weighted_mse":
            thresholds = self._thresholds.to(targets.device)
            weights = self._quantile_weights.to(targets.device)
            bin_indices = torch.bucketize(targets.contiguous(), thresholds, right=False)
            squared_error = weights[bin_indices] * squared_error
        return torch.mean(squared_error)

    @staticmethod
    def _metric_values(
        targets: torch.Tensor,
        predictions: torch.Tensor,
    ) -> dict[str, float]:
        residuals = targets - predictions
        mse = torch.mean(torch.square(residuals))
        mae = torch.mean(torch.abs(residuals))
        total = torch.sum(torch.square(targets - torch.mean(targets)))
        r2 = 1.0 - torch.sum(torch.square(residuals)) / total if total > 0 else 0.0
        return {"mse": float(mse), "mae": float(mae), "r2": float(r2)}

    def _as_tensors(
        self,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        X = torch.as_tensor(np.asarray(features), dtype=torch.float32)
        y = torch.as_tensor(np.asarray(targets), dtype=torch.float32)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return X, y

    def _evaluate_tensors(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(features.to(self.device))
            device_targets = targets.to(self.device)
            values = self._metric_values(device_targets, predictions)
            values["loss"] = float(self._loss(device_targets, predictions))
        return values

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 0,
        early_stopping: bool = True,
        patience: int = 10,
        early_stopping_metric: str = "loss",
        warm_up: int = 0,
    ) -> dict[str, list[float]]:
        """Train the model and return a Keras-compatible history mapping."""
        monitor = str(early_stopping_metric).strip().lower()
        if monitor not in SUPPORTED_EARLY_STOPPING_METRICS:
            supported = ", ".join(SUPPORTED_EARLY_STOPPING_METRICS)
            raise ValueError(
                f"Unsupported early_stopping_metric: {monitor!r}. Use one of: {supported}"
            )
        if warm_up < 0:
            raise ValueError("warm_up must be a non-negative integer.")

        train_X, train_y = self._as_tensors(X_train, y_train)
        validation = X_val is not None and y_val is not None
        if validation:
            val_X, val_y = self._as_tensors(X_val, y_val)
        generator = torch.Generator().manual_seed(self.random_state)
        loader = DataLoader(
            TensorDataset(train_X, train_y),
            batch_size=int(batch_size),
            shuffle=True,
            generator=generator,
        )
        metric_names = ["loss", "mae", "mse", "r2"]
        history = {name: [] for name in metric_names}
        if validation:
            history.update({f"val_{name}": [] for name in metric_names})

        maximize = monitor == "r2"
        best_value = -np.inf if maximize else np.inf
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(int(epochs)):
            self.model.train()
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                predictions = self.model(batch_X)
                loss = self._loss(batch_y, predictions)
                loss.backward()
                self.optimizer.step()

            train_values = self._evaluate_tensors(train_X, train_y)
            for name in metric_names:
                history[name].append(train_values[name])
            val_values = None
            if validation:
                val_values = self._evaluate_tensors(val_X, val_y)
                for name in metric_names:
                    history[f"val_{name}"].append(val_values[name])

            if verbose:
                message = f"Epoch {epoch + 1}/{epochs} - loss: {train_values['loss']:.4f}"
                if val_values is not None:
                    message += f" - val_loss: {val_values['loss']:.4f}"
                print(message, flush=True)

            if early_stopping and val_values is not None and epoch >= int(warm_up):
                current = val_values[monitor]
                improved = current > best_value if maximize else current < best_value
                if improved:
                    best_value = current
                    best_state = deepcopy(self.model.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= int(patience):
                        if verbose:
                            print(f"Early stopping at epoch {epoch + 1}", flush=True)
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.history = history
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions as a two-dimensional NumPy array."""
        features = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(features.to(self.device))
        return predictions.detach().cpu().numpy()

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        """Evaluate loss and standard regression metrics."""
        features, targets = self._as_tensors(X_test, y_test)
        values = self._evaluate_tensors(features, targets)
        values["rmse"] = float(np.sqrt(values["mse"]))
        return values

    def get_model_summary(self) -> str:
        """Return a readable architecture and device summary."""
        return f"{self.model}\nDevice: {self.device}"
