Computes the logarithm of the hyperbolic cosine of the prediction error.

Formula:

```python
error = y_pred - y_true
logcosh = mean(log((exp(error) + exp(-error))/2), axis=-1)
```

Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

Examples:

Standalone usage:

>>> m = keras.metrics.LogCoshError()
>>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
>>> m.result()
0.10844523
>>> m.reset_state()
>>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
...                sample_weight=[1, 0])
>>> m.result()
0.21689045

Usage with `compile()` API:

```python
model.compile(optimizer='sgd',
              loss='mse',
              metrics=[keras.metrics.LogCoshError()])
```
