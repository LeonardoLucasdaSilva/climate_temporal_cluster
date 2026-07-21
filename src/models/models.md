# Models

This package contains trainable model implementations.

- `lstm.py`: TensorFlow/Keras LSTM model for sequence-to-precipitation
  prediction, with one or two recurrent layers, configurable output width for
  multi-lead-day targets, and an AdamW optimizer with configurable decoupled
  weight decay. It also provides
  `weighted_mse_loss(y_real, y_pred, alpha)`, which returns the per-sample mean
  of `(1 + alpha * y_real) * |y_real - y_pred|^2` for a finite `alpha > 0`.

Typical usage:

```python
from models.lstm import LSTMPrecipitationPredictor

model = LSTMPrecipitationPredictor(
    input_shape=(1, n_features),
    lstm_units_2=None,  # Use one LSTM layer instead of two
    output_units=forecast_horizon,
    weight_decay=1e-4,
    loss_function="weighted_mse_loss",
    loss_alpha=1.0,
)
history = model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
```
