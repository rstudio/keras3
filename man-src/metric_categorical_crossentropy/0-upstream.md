keras.metrics.CategoricalCrossentropy
__signature__
(
  name='categorical_crossentropy',
  dtype=None,
  from_logits=False,
  label_smoothing=0,
  axis=-1
)
__doc__
Computes the crossentropy metric between the labels and predictions.

This is the crossentropy metric class to be used when there are multiple
label classes (2 or more). It assumes that labels are one-hot encoded,
e.g., when labels values are `[2, 0, 1]`, then
`y_true` is `[[0, 0, 1], [1, 0, 0], [0, 1, 0]]`.

Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.
    from_logits: (Optional) Whether output is expected to be
        a logits tensor. By default, we consider that output
        encodes a probability distribution.
    label_smoothing: (Optional) Float in `[0, 1]`.
        When > 0, label values are smoothed, meaning the confidence
        on label values are relaxed. e.g. `label_smoothing=0.2` means
        that we will use a value of 0.1 for label
        "0" and 0.9 for label "1".
    axis: (Optional) Defaults to `-1`.
        The dimension along which entropy is computed.

Examples:

Standalone usage:

>>> # EPSILON = 1e-7, y = y_true, y` = y_pred
>>> # y` = clip_ops.clip_by_value(output, EPSILON, 1. - EPSILON)
>>> # y` = [[0.05, 0.95, EPSILON], [0.1, 0.8, 0.1]]
>>> # xent = -sum(y * log(y'), axis = -1)
>>> #      = -((log 0.95), (log 0.1))
>>> #      = [0.051, 2.302]
>>> # Reduced xent = (0.051 + 2.302) / 2
>>> m = keras.metrics.CategoricalCrossentropy()
>>> m.update_state([[0, 1, 0], [0, 0, 1]],
...                [[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
>>> m.result()
1.1769392

>>> m.reset_state()
>>> m.update_state([[0, 1, 0], [0, 0, 1]],
...                [[0.05, 0.95, 0], [0.1, 0.8, 0.1]],
...                sample_weight=np.array([0.3, 0.7]))
>>> m.result()
1.6271976

Usage with `compile()` API:

```python
model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=[keras.metrics.CategoricalCrossentropy()])
```
