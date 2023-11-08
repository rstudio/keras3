keras.losses.Huber
__signature__
(delta=1.0, reduction='sum_over_batch_size', name='huber_loss')
__doc__
Computes the Huber loss between `y_true` & `y_pred`.

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

Args:
    delta: A float, the point where the Huber loss function changes from a
        quadratic to linear.
    reduction: Type of reduction to apply to loss. Options are `"sum"`,
        `"sum_over_batch_size"` or `None`. Defaults to
        `"sum_over_batch_size"`.
    name: Optional name for the instance.
