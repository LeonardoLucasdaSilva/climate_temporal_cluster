"""LSTM model for sequence-to-value precipitation prediction."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers


class LSTMPrecipitationPredictor:
    """LSTM model for predicting next-day precipitation from window features.

    This model takes cluster-assigned window features as input and predicts
    the precipitation value for the day following the window.

    Architecture:
    - Input: (sequence_length, n_features) - typically flattened window features
    - Two LSTM layers with dropout for regularization
    - Dense layers for feature transformation
    - Output: Single value (precipitation prediction)
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        lstm_units: int = 64,
        lstm_units_2: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        random_state: int = 42,
    ):
        """Initialize LSTM model.

        Args:
            input_shape: Shape of input data (sequence_length, n_features) or (n_features,)
            lstm_units: Number of units in first LSTM layer
            lstm_units_2: Number of units in second LSTM layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            random_state: Random seed for reproducibility
        """
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.lstm_units_2 = lstm_units_2
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.random_state = random_state
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

            # Output layer (precipitation value)
            layers.Dense(1, activation='linear'),  # Linear activation for regression
        ])

        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
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
            y_train: Training target values
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
            Predicted precipitation values
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
            y_test: Test target values

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

