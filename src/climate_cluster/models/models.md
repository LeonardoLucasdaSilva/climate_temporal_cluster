# Models

This package contains trainable model implementations.

- `lstm_model.py`: TensorFlow/Keras LSTM model for sequence-to-value precipitation
  prediction.

Typical usage:

```python
from climate_cluster.models.lstm_model import LSTMPrecipitationPredictor

model = LSTMPrecipitationPredictor(input_shape=(1, n_features))
history = model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
```
