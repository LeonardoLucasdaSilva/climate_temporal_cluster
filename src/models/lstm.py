"""LSTM model for sequence-to-precipitation prediction."""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers


SUPPORTED_LOSS_FUNCTIONS = (
    "mean_squared_error",
    "mse",
    "mean_absolute_error",
    "mae",
    "huber",
    "quantile_weighted_mse",
)


def quantile_weighted_mse_loss(
    thresholds_mm: Sequence[float],
    weights: Sequence[float],
) -> Callable:
    """Return MSE weighted by target bins separated by precipitation thresholds.

    Args:
        thresholds_mm: Increasing precipitation thresholds in millimeters.
            With `n` thresholds, the loss has `n + 1` target bins.
        weights: Positive weight for each target bin. Must contain exactly
            `len(thresholds_mm) + 1` values.

    Returns:
        A Keras-compatible loss function.
    """
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

    thresholds_tf = tf.constant(thresholds, dtype=tf.float32)
    weights_tf = tf.constant(bin_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true_float = tf.cast(y_true, tf.float32)
        y_pred_float = tf.cast(y_pred, tf.float32)
        bin_indices = tf.reduce_sum(
            tf.cast(y_true_float[..., tf.newaxis] > thresholds_tf, tf.int32),
            axis=-1,
        )
        sample_weights = tf.gather(weights_tf, bin_indices)
        return tf.reduce_mean(sample_weights * tf.square(y_true_float - y_pred_float))

    loss.__name__ = "quantile_weighted_mse"
    return loss


def resolve_loss_function(
    loss_function: str,
    quantile_thresholds_mm: Sequence[float] | None = None,
    quantile_weights: Sequence[float] | None = None,
) -> str | Callable:
    """Return a Keras loss from a configured loss name and optional parameters."""
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


class LSTMPrecipitationPredictor:
    """LSTM model for predicting precipitation targets from window features.

    This model takes cluster-assigned window features as input and predicts
    one or more precipitation values after the window.

    Architecture:
    - Input: (sequence_length, n_features) - typically flattened window features
    - Two LSTM layers with dropout for regularization
    - Dense layers for feature transformation
    - Output: One value per configured forecast lead day
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
        output_units: int = 1,
    ):
        """Initialize LSTM model.

        Args:
            input_shape: Shape of input data (sequence_length, n_features) or (n_features,)
            lstm_units: Number of units in first LSTM layer
            lstm_units_2: Number of units in second LSTM layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            random_state: Random seed for reproducibility
            loss_function: Keras loss name or `quantile_weighted_mse`.
            loss_quantile_thresholds_mm: Cluster-specific rain thresholds used
                by `quantile_weighted_mse`.
            loss_quantile_weights: Target-bin weights used by
                `quantile_weighted_mse`.
            output_units: Number of precipitation target columns to predict.
        """
        if output_units <= 0:
            raise ValueError("output_units must be positive.")
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.lstm_units_2 = lstm_units_2
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.loss_function = loss_function
        self.loss_quantile_thresholds_mm = loss_quantile_thresholds_mm
        self.loss_quantile_weights = loss_quantile_weights
        self.output_units = output_units
        self.history = None
        self.model = None

        # Set random seeds for reproducibility
        np.random.seed(random_state)
        tf.random.set_seed(random_state)

        self._build_model()

    def _build_model(self):
        """Build the LSTM model architecture."""
        model = keras.Sequential([
            # First LSTM layer
            layers.LSTM(
                self.lstm_units,
                activation='relu',
                input_shape=self.input_shape,
                return_sequences=True,  # Return sequences for second LSTM
            ),
            layers.Dropout(self.dropout_rate),

            # Second LSTM layer
            layers.LSTM(
                self.lstm_units_2,
                activation='relu',
                return_sequences=False,  # Return only last output
            ),
            layers.Dropout(self.dropout_rate),

            # Dense layers for final prediction
            layers.Dense(16, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(8, activation='relu'),

            # Output layer (one precipitation value per lead day)
            layers.Dense(
                self.output_units,
                activation='linear',
            ),
        ])

        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = resolve_loss_function(
            self.loss_function,
            quantile_thresholds_mm=self.loss_quantile_thresholds_mm,
            quantile_weights=self.loss_quantile_weights,
        )
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['mae', 'mse'],
        )

        self.model = model

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
    ) -> keras.callbacks.History:
        """Train the LSTM model.

        Args:
            X_train: Training features
            y_train: Training target values, either one column or one column per
                forecast lead day
            X_val: Validation features (optional)
            y_val: Validation target values (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level (0, 1, or 2)
            early_stopping: Whether to use early stopping
            patience: Patience for early stopping

        Returns:
            History object with training metrics
        """
        validation_data = None
        callbacks = []

        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        if early_stopping and validation_data is not None:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True,
                    verbose=verbose,
                )
            )

        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
        )

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.

        Args:
            X: Input features

        Returns:
            Predicted precipitation values, one column per configured output
        """
        if self.model is None:
            raise RuntimeError("Model has not been built. Call fit() first.")

        return self.model.predict(X, verbose=0)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        """Evaluate model on test data.

        Args:
            X_test: Test features
            y_test: Test target values, either one column or one column per
                forecast lead day

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise RuntimeError("Model has not been built. Call fit() first.")

        loss, mae, mse = self.model.evaluate(X_test, y_test, verbose=0)
        rmse = np.sqrt(mse)

        return {
            'loss': loss,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
        }

    def get_model_summary(self) -> str:
        """Get model architecture summary.

        Returns:
            String representation of model architecture
        """
        if self.model is None:
            raise RuntimeError("Model has not been built.")

        summary_str = []
        self.model.summary(print_fn=lambda x: summary_str.append(x))
        return '\n'.join(summary_str)


def prepare_sequences(
    X: np.ndarray,
    sequence_length: int = 1,
) -> np.ndarray:
    """Prepare data for LSTM by creating sequences.

    If input is already 2D (samples, features), returns as-is or reshapes
    to (samples, sequence_length, features) if sequence_length > 1.

    Args:
        X: Input array of shape (n_samples, n_features) or already shaped for LSTM
        sequence_length: Number of time steps per sequence

    Returns:
        Array ready for LSTM input
    """
    if len(X.shape) == 2:
        n_samples, n_features = X.shape
        if sequence_length == 1:
            # Keep as 2D, LSTM will interpret as (samples, n_features, 1)
            return X.reshape(n_samples, n_features, 1)
        else:
            # Need more complex sequence creation
            # This is a simplified version - full implementation would create
            # overlapping sequences
            return X.reshape(n_samples, 1, n_features)

    return X

