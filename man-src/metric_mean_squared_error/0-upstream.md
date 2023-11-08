keras.metrics.MeanSquaredError
__signature__
(name='mean_squared_error', dtype=None)
__doc__
Computes the mean squared error between `y_true` and `y_pred`.

Formula:

```python
loss = mean(square(y_true - y_pred))
```

Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

Example:

>>> m = keras.metrics.MeanSquaredError()
>>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
>>> m.result()
0.25
