Computes the Huber loss between `y_true` & `y_pred`.

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

@returns
    Tensor with one scalar loss entry per sample.

@param delta
A float, the point where the Huber loss function changes from a
quadratic to linear. Defaults to `1.0`.

@param reduction
Type of reduction to apply to loss. Options are `"sum"`,
`"sum_over_batch_size"` or `None`. Defaults to
`"sum_over_batch_size"`.

@param name
Optional name for the instance.

@param y_true
tensor of true targets.

@param y_pred
tensor of predicted targets.

@param ...
Passed on to the Python callable

@export
@family losses
@seealso
+ <https:/keras.io/api/losses/regression_losses#huber-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/losses/Huber>
