Computes the Poisson metric between `y_true` and `y_pred`.

Formula:

```python
metric = y_pred - y_true * log(y_pred)
```

Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

Examples:

Standalone usage:

>>> m = keras.metrics.Poisson()
>>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
>>> m.result()
0.49999997

>>> m.reset_state()
>>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
...                sample_weight=[1, 0])
>>> m.result()
0.99999994

Usage with `compile()` API:

```python
model.compile(optimizer='sgd',
              loss='mse',
              metrics=[keras.metrics.Poisson()])
```
