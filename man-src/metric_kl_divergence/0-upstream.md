keras.metrics.KLDivergence
__signature__
(name='kl_divergence', dtype=None)
__doc__
Computes Kullback-Leibler divergence metric between `y_true` and
`y_pred`.

Formula:

```python
metric = y_true * log(y_true / y_pred)
```

Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

Usage:

Standalone usage:

>>> m = keras.metrics.KLDivergence()
>>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
>>> m.result()
0.45814306

>>> m.reset_state()
>>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
...                sample_weight=[1, 0])
>>> m.result()
0.9162892

Usage with `compile()` API:

```python
model.compile(optimizer='sgd',
              loss='mse',
              metrics=[keras.metrics.KLDivergence()])
```
