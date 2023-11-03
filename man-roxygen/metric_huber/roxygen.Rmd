Computes Huber loss value.

@description
Formula:
```python
for x in error:
    if abs(x) <= delta:
        loss.append(0.5 * x^2)
    elif abs(x) > delta:
        loss.append(delta * abs(x) - 0.5 * delta^2)

loss = mean(loss, axis=-1)
```
See: [Huber loss](https://en.wikipedia.org/wiki/Huber_loss).

# Examples
```python
y_true = [[0, 1], [0, 0]]
y_pred = [[0.6, 0.4], [0.4, 0.6]]
loss = keras.losses.huber(y_true, y_pred)
# 0.155
```

# Returns
    Tensor with one scalar loss entry per sample.

@param y_true tensor of true targets.
@param y_pred tensor of predicted targets.
@param delta A float, the point where the Huber loss function changes from a
    quadratic to linear. Defaults to `1.0`.

@export
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/huber>
