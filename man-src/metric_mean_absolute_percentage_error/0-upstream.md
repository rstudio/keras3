keras.metrics.MeanAbsolutePercentageError
__signature__
(name='mean_absolute_percentage_error', dtype=None)
__doc__
Computes mean absolute percentage error between `y_true` and `y_pred`.

Formula:

```python
loss = 100 * mean(abs((y_true - y_pred) / y_true))
```

Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

Examples:

Standalone usage:

>>> m = keras.metrics.MeanAbsolutePercentageError()
>>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
>>> m.result()
250000000.0
>>> m.reset_state()
>>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
...                sample_weight=[1, 0])
>>> m.result()
500000000.0

Usage with `compile()` API:

```python
model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=[keras.metrics.MeanAbsolutePercentageError()])
```
