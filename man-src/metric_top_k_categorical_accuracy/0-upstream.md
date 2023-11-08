keras.metrics.TopKCategoricalAccuracy
__signature__
(k=5, name='top_k_categorical_accuracy', dtype=None)
__doc__
Computes how often targets are in the top `K` predictions.

Args:
    k: (Optional) Number of top elements to look at for computing accuracy.
        Defaults to `5`.
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

Usage:

Standalone usage:

>>> m = keras.metrics.TopKCategoricalAccuracy(k=1)
>>> m.update_state([[0, 0, 1], [0, 1, 0]],
...                [[0.1, 0.9, 0.8], [0.05, 0.95, 0]])
>>> m.result()
0.5

>>> m.reset_state()
>>> m.update_state([[0, 0, 1], [0, 1, 0]],
...                [[0.1, 0.9, 0.8], [0.05, 0.95, 0]],
...                sample_weight=[0.7, 0.3])
>>> m.result()
0.3

Usage with `compile()` API:

```python
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=[keras.metrics.TopKCategoricalAccuracy()])
```
