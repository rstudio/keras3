keras.losses.MeanSquaredLogarithmicError
__signature__
(reduction='sum_over_batch_size', name='mean_squared_logarithmic_error')
__doc__
Computes the mean squared logarithmic error between `y_true` & `y_pred`.

Formula:

```python
loss = mean(square(log(y_true + 1) - log(y_pred + 1)))
```

Args:
    reduction: Type of reduction to apply to the loss. In almost all cases
        this should be `"sum_over_batch_size"`.
        Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
    name: Optional name for the loss instance.
