keras.losses.Hinge
__signature__
(reduction='sum_over_batch_size', name='hinge')
__doc__
Computes the hinge loss between `y_true` & `y_pred`.

Formula:

```python
loss = maximum(1 - y_true * y_pred, 0)
```

`y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
provided we will convert them to -1 or 1.

Args:
    reduction: Type of reduction to apply to the loss. In almost all cases
        this should be `"sum_over_batch_size"`.
        Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
    name: Optional name for the loss instance.
