keras.metrics.MeanSquaredLogarithmicError
__signature__
(name='mean_squared_logarithmic_error', dtype=None)
__doc__
Computes mean squared logarithmic error between `y_true` and `y_pred`.

Formula:

```python
loss = mean(square(log(y_true + 1) - log(y_pred + 1)))
```

Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

Examples:

Standalone usage:

>>> m = keras.metrics.MeanSquaredLogarithmicError()
>>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
>>> m.result()
0.12011322
>>> m.reset_state()
>>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
...                sample_weight=[1, 0])
>>> m.result()
0.24022643

Usage with `compile()` API:

```python
model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=[keras.metrics.MeanSquaredLogarithmicError()])
```
