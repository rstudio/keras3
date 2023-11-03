Computes the crossentropy metric between the labels and predictions.

This is the crossentropy metric class to be used when there are only two
label classes (0 and 1).

Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.
    from_logits: (Optional) Whether output is expected
        to be a logits tensor. By default, we consider
        that output encodes a probability distribution.
    label_smoothing: (Optional) Float in `[0, 1]`.
        When > 0, label values are smoothed,
        meaning the confidence on label values are relaxed.
        e.g. `label_smoothing=0.2` means that we will use
        a value of 0.1 for label "0" and 0.9 for label "1".

Examples:

Standalone usage:

>>> m = keras.metrics.BinaryCrossentropy()
>>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
>>> m.result()
0.81492424

>>> m.reset_state()
>>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
...                sample_weight=[1, 0])
>>> m.result()
0.9162905

Usage with `compile()` API:

```python
model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=[keras.metrics.BinaryCrossentropy()])
```
