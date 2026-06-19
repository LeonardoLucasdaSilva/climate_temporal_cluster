# Models

This package contains trainable model implementations.

- `lstm.py`: TensorFlow/Keras LSTM model for sequence-to-value precipitation
  prediction.

Typical usage:

```python
from models.lstm import LSTMPrecipitationPredictor

model = LSTMPrecipitationPredictor(input_shape=(1, n_features))
history = model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
```
