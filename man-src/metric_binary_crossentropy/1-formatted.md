Computes the crossentropy metric between the labels and predictions.

@description
This is the crossentropy metric class to be used when there are only two
label classes (0 and 1).

# Examples
```python
y_true = [[0, 1], [0, 0]]
y_pred = [[0.6, 0.4], [0.4, 0.6]]
loss = keras.losses.binary_crossentropy(y_true, y_pred)
assert loss.shape == (2,)
loss
# array([0.916 , 0.714], dtype=float32)
```
Standalone usage:

```python
m = keras.metrics.BinaryCrossentropy()
m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
m.result()
# 0.81492424
```

```python
m.reset_state()
m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
               sample_weight=[1, 0])
m.result()
# 0.9162905
```

Usage with `compile()` API:

```python
model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=[keras.metrics.BinaryCrossentropy()])
```

@returns
Binary crossentropy loss value. shape = `[batch_size, d0, .. dN-1]`.

@param name (Optional) string name of the metric instance.
@param dtype (Optional) data type of the metric result.
@param from_logits (Optional) Whether output is expected
    to be a logits tensor. By default, we consider
    that output encodes a probability distribution.
@param label_smoothing (Optional) Float in `[0, 1]`.
    When > 0, label values are smoothed,
    meaning the confidence on label values are relaxed.
    e.g. `label_smoothing=0.2` means that we will use
    a value of 0.1 for label "0" and 0.9 for label "1".
@param y_true Ground truth values. shape = `[batch_size, d0, .. dN]`.
@param y_pred The predicted values. shape = `[batch_size, d0, .. dN]`.
@param axis The axis along which the mean is computed. Defaults to `-1`.
@param ... Passed on to the Python callable

@export
@family metric
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/BinaryCrossentropy>
