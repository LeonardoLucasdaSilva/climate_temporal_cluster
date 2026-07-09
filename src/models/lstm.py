"""PyTorch LSTM model for sequence-to-value precipitation prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SUPPORTED_LOSS_FUNCTIONS = (
    "mean_squared_error",
    "mse",
    "mean_absolute_error",
    "mae",
    "huber",
    "quantile_weighted_mse",
)


@dataclass
class TrainingHistory:
    """Keras-like training history used by existing plotting helpers."""

    history: dict[str, list[float]]


def quantile_weighted_mse_loss(
    thresholds_mm: Sequence[float],
    weights: Sequence[float],
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return MSE weighted by target bins separated by precipitation thresholds."""
    thresholds = np.asarray(thresholds_mm, dtype=np.float32)
    bin_weights = np.asarray(weights, dtype=np.float32)
    if thresholds.ndim != 1:
        raise ValueError("thresholds_mm must be one-dimensional.")
    if bin_weights.ndim != 1:
        raise ValueError("weights must be one-dimensional.")
    if len(bin_weights) != len(thresholds) + 1:
        raise ValueError(
            "weights must contain exactly len(thresholds_mm) + 1 values."
        )
    if len(thresholds) and np.any(np.diff(thresholds) <= 0):
        raise ValueError("thresholds_mm must be strictly increasing.")
    if np.any(bin_weights <= 0):
        raise ValueError("weights must be positive.")

    thresholds_tensor = torch.tensor(thresholds, dtype=torch.float32)
    weights_tensor = torch.tensor(bin_weights, dtype=torch.float32)

    def loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        thresholds_device = thresholds_tensor.to(y_true.device)
        weights_device = weights_tensor.to(y_true.device)
        y_true_float = y_true.float()
        y_pred_float = y_pred.float()
        bin_indices = (y_true_float.unsqueeze(-1) > thresholds_device).sum(dim=-1)
        sample_weights = weights_device[bin_indices]
        return torch.mean(sample_weights * torch.square(y_true_float - y_pred_float))

    loss.__name__ = "quantile_weighted_mse"
    return loss


def resolve_loss_function(
    loss_function: str,
    quantile_thresholds_mm: Sequence[float] | None = None,
    quantile_weights: Sequence[float] | None = None,
) -> str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return a PyTorch loss from a configured loss name and optional parameters."""
    normalized_loss = loss_function.lower()
    if normalized_loss not in SUPPORTED_LOSS_FUNCTIONS:
        supported = ", ".join(SUPPORTED_LOSS_FUNCTIONS)
        raise ValueError(
            f"Unsupported loss_function: {loss_function!r}. Use one of: {supported}"
        )
    if normalized_loss == "quantile_weighted_mse":
        if quantile_thresholds_mm is None or quantile_weights is None:
            raise ValueError(
                "quantile_weighted_mse requires thresholds and weights."
            )
        return quantile_weighted_mse_loss(
            quantile_thresholds_mm,
            quantile_weights,
        )
    return normalized_loss


class KerasStyleLSTMLayer(nn.Module):
    """Single LSTM layer with Keras-compatible ReLU cell activation."""

    def __init__(self, input_size: int, hidden_size: int, return_sequences: bool):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences
        self.kernel = nn.Parameter(torch.empty(input_size, 4 * hidden_size))
        self.recurrent_kernel = nn.Parameter(torch.empty(hidden_size, 4 * hidden_size))
        self.bias = nn.Parameter(torch.empty(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.kernel)
        nn.init.orthogonal_(self.recurrent_kernel)
        nn.init.zeros_(self.bias)
        with torch.no_grad():
            self.bias[self.hidden_size : 2 * self.hidden_size].fill_(1.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, timesteps, _ = inputs.shape
        h_t = inputs.new_zeros(batch_size, self.hidden_size)
        c_t = inputs.new_zeros(batch_size, self.hidden_size)
        outputs = []

        for timestep in range(timesteps):
            gates = (
                inputs[:, timestep, :] @ self.kernel
                + h_t @ self.recurrent_kernel
                + self.bias
            )
            input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, dim=1)
            input_gate = torch.sigmoid(input_gate)
            forget_gate = torch.sigmoid(forget_gate)
            cell_gate = torch.relu(cell_gate)
            output_gate = torch.sigmoid(output_gate)
            c_t = forget_gate * c_t + input_gate * cell_gate
            h_t = output_gate * torch.relu(c_t)
            outputs.append(h_t.unsqueeze(1))

        sequence = torch.cat(outputs, dim=1)
        if self.return_sequences:
            return sequence
        return sequence[:, -1, :]


class LSTMPrecipitationNet(nn.Module):
    """Two-layer LSTM regression network matching the previous Keras topology."""

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        lstm_units: int,
        lstm_units_2: int,
        dropout_rate: float,
    ):
        super().__init__()
        if len(input_shape) != 2:
            raise ValueError("input_shape must be (sequence_length, n_features).")
        _, n_features = input_shape
        self.layers = nn.Sequential(
            KerasStyleLSTMLayer(n_features, lstm_units, return_sequences=True),
            nn.Dropout(dropout_rate),
            KerasStyleLSTMLayer(lstm_units, lstm_units_2, return_sequences=False),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_units_2, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class LSTMPrecipitationPredictor:
    """LSTM model for predicting next-day precipitation from window features.

    The public API mirrors the previous TensorFlow/Keras wrapper while using
    PyTorch tensors, modules, optimizers, and training loops internally.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        lstm_units: int = 64,
        lstm_units_2: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        random_state: int = 42,
        loss_function: str = "mean_squared_error",
        loss_quantile_thresholds_mm: Sequence[float] | None = None,
        loss_quantile_weights: Sequence[float] | None = None,
        device: str | torch.device | None = None,
        max_grad_norm: float = 1.0,
    ):
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.lstm_units_2 = lstm_units_2
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.loss_function = loss_function
        self.loss_quantile_thresholds_mm = loss_quantile_thresholds_mm
        self.loss_quantile_weights = loss_quantile_weights
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_grad_norm = max_grad_norm
        self.history: TrainingHistory | None = None
        self.model: LSTMPrecipitationNet | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None

        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)

        self._build_model()

    def _build_model(self) -> None:
        """Build the PyTorch LSTM model architecture."""
        self.model = LSTMPrecipitationNet(
            input_shape=self.input_shape,
            lstm_units=self.lstm_units,
            lstm_units_2=self.lstm_units_2,
            dropout_rate=self.dropout_rate,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = self._make_criterion(
            resolve_loss_function(
                self.loss_function,
                quantile_thresholds_mm=self.loss_quantile_thresholds_mm,
                quantile_weights=self.loss_quantile_weights,
            )
        )

    def _make_criterion(
        self,
        loss: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        if callable(loss):
            return loss
        if loss in ("mean_squared_error", "mse"):
            return nn.MSELoss()
        if loss in ("mean_absolute_error", "mae"):
            return nn.L1Loss()
        if loss == "huber":
            return nn.HuberLoss(delta=1.0)
        raise ValueError(f"Unsupported loss_function: {loss!r}")

    def _as_tensor(self, values: np.ndarray) -> torch.Tensor:
        array = np.asarray(values, dtype=np.float32)
        array = np.nan_to_num(array, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def _prepare_targets(self, values: np.ndarray) -> torch.Tensor:
        tensor = self._as_tensor(values)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(1)
        return tensor

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
    ) -> TrainingHistory:
        """Train the LSTM model and return Keras-like metric history."""
        if self.model is None or self.optimizer is None or self.criterion is None:
            raise RuntimeError("Model has not been built.")

        X_train_tensor = self._as_tensor(X_train)
        y_train_tensor = self._prepare_targets(y_train)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        generator = torch.Generator()
        generator.manual_seed(self.random_state)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )

        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (self._as_tensor(X_val), self._prepare_targets(y_val))

        history: dict[str, list[float]] = {"loss": [], "mae": [], "mse": []}
        if validation_data is not None:
            history.update({"val_loss": [], "val_mae": [], "val_mse": []})

        best_state = None
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            total_mae = 0.0
            total_mse = 0.0
            total_samples = 0

            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.criterion(batch_y, predictions)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                predictions_for_metrics = torch.nan_to_num(
                    predictions.detach(), nan=0.0, posinf=1.0e6, neginf=-1.0e6
                )

                batch_size_actual = batch_X.shape[0]
                total_loss += loss.detach().item() * batch_size_actual
                total_mae += torch.mean(torch.abs(batch_y - predictions_for_metrics)).item() * batch_size_actual
                total_mse += torch.mean(torch.square(batch_y - predictions_for_metrics)).item() * batch_size_actual
                total_samples += batch_size_actual

            if total_samples == 0:
                raise FloatingPointError(
                    "All training batches produced non-finite loss. Check input scaling, "
                    "target values, or loss weights."
                )

            history["loss"].append(total_loss / total_samples)
            history["mae"].append(total_mae / total_samples)
            history["mse"].append(total_mse / total_samples)

            if validation_data is not None:
                val_loss, val_mae, val_mse = self._evaluate_tensors(*validation_data)
                history["val_loss"].append(val_loss)
                history["val_mae"].append(val_mae)
                history["val_mse"].append(val_mse)
                monitored_loss = val_loss
            else:
                monitored_loss = history["loss"][-1]

            if verbose:
                message = (
                    f"Epoch {epoch + 1}/{epochs} - loss: {history['loss'][-1]:.4f} "
                    f"- mae: {history['mae'][-1]:.4f} - mse: {history['mse'][-1]:.4f}"
                )
                if validation_data is not None:
                    message += (
                        f" - val_loss: {history['val_loss'][-1]:.4f}"
                        f" - val_mae: {history['val_mae'][-1]:.4f}"
                        f" - val_mse: {history['val_mse'][-1]:.4f}"
                    )
                print(message)

            if early_stopping and validation_data is not None:
                if np.isfinite(monitored_loss) and monitored_loss < best_val_loss:
                    best_val_loss = monitored_loss
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch + 1}.")
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.history = TrainingHistory(history=history)
        return self.history

    def _evaluate_tensors(
        self,
        X_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
    ) -> tuple[float, float, float]:
        if self.model is None or self.criterion is None:
            raise RuntimeError("Model has not been built.")
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
            predictions_for_metrics = torch.nan_to_num(
                predictions, nan=0.0, posinf=1.0e6, neginf=-1.0e6
            )
            loss_tensor = self.criterion(y_tensor, predictions_for_metrics)
            loss = loss_tensor.item() if torch.isfinite(loss_tensor) else float("inf")
            mae = torch.mean(torch.abs(y_tensor - predictions_for_metrics)).item()
            mse = torch.mean(torch.square(y_tensor - predictions_for_metrics)).item()
        return loss, mae, mse

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        if self.model is None:
            raise RuntimeError("Model has not been built. Call fit() first.")

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(self._as_tensor(X))
            predictions = torch.nan_to_num(predictions, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        return predictions.detach().cpu().numpy()

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        """Evaluate model on test data."""
        loss, mae, mse = self._evaluate_tensors(
            self._as_tensor(X_test),
            self._prepare_targets(y_test),
        )
        rmse = np.sqrt(mse)

        return {
            "loss": loss,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
        }

    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        if self.model is None:
            raise RuntimeError("Model has not been built.")

        lines = [str(self.model)]
        trainable_params = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        total_params = sum(parameter.numel() for parameter in self.model.parameters())
        lines.append(f"Total params: {total_params:,}")
        lines.append(f"Trainable params: {trainable_params:,}")
        lines.append(f"Device: {self.device}")
        return "\n".join(lines)


def prepare_sequences(
    X: np.ndarray,
    sequence_length: int = 1,
) -> np.ndarray:
    """Prepare data for LSTM by creating sequences."""
    if len(X.shape) == 2:
        n_samples, n_features = X.shape
        if sequence_length == 1:
            return X.reshape(n_samples, n_features, 1)
        return X.reshape(n_samples, 1, n_features)

    return X
