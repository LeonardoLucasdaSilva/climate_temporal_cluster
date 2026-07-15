# Models

This package contains trainable model implementations.

- `lstm.py`: TensorFlow/Keras LSTM model for sequence-to-precipitation
  prediction, with configurable output width for multi-lead-day targets and an
  AdamW optimizer with configurable decoupled weight decay.

Typical usage:

```python
from models.lstm import LSTMPrecipitationPredictor

model = LSTMPrecipitationPredictor(
    input_shape=(1, n_features),
    output_units=forecast_horizon,
    weight_decay=1e-4,
)
history = model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
```
