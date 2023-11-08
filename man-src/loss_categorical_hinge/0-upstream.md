keras.losses.CategoricalHinge
__signature__
(reduction='sum_over_batch_size', name='categorical_hinge')
__doc__
Computes the categorical hinge loss between `y_true` & `y_pred`.

Formula:

```python
loss = maximum(neg - pos + 1, 0)
```

where `neg=maximum((1-y_true)*y_pred)` and `pos=sum(y_true*y_pred)`

Args:
    reduction: Type of reduction to apply to the loss. In almost all cases
        this should be `"sum_over_batch_size"`.
        Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
    name: Optional name for the loss instance.
