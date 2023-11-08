keras.metrics.huber
__signature__
(y_true, y_pred, delta=1.0)
__doc__
Computes Huber loss value.

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

Example:

>>> y_true = [[0, 1], [0, 0]]
>>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
>>> loss = keras.losses.huber(y_true, y_pred)
0.155


Args:
    y_true: tensor of true targets.
    y_pred: tensor of predicted targets.
    delta: A float, the point where the Huber loss function changes from a
        quadratic to linear. Defaults to `1.0`.

Returns:
    Tensor with one scalar loss entry per sample.
