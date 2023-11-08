keras.metrics.MeanAbsoluteError
__signature__
(name='mean_absolute_error', dtype=None)
__doc__
Computes the mean absolute error between the labels and predictions.

Formula:

```python
loss = mean(abs(y_true - y_pred))
```

Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

Examples:

Standalone usage:

>>> m = keras.metrics.MeanAbsoluteError()
>>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
>>> m.result()
0.25
>>> m.reset_state()
>>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
...                sample_weight=[1, 0])
>>> m.result()
0.5

Usage with `compile()` API:

```python
model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=[keras.metrics.MeanAbsoluteError()])
```
