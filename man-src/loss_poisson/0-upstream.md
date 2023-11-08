keras.losses.Poisson
__signature__
(reduction='sum_over_batch_size', name='poisson')
__doc__
Computes the Poisson loss between `y_true` & `y_pred`.

Formula:

```python
loss = y_pred - y_true * log(y_pred)
```

Args:
    reduction: Type of reduction to apply to the loss. In almost all cases
        this should be `"sum_over_batch_size"`.
        Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
    name: Optional name for the loss instance.
