keras.losses.MeanSquaredError
__signature__
(reduction='sum_over_batch_size', name='mean_squared_error')
__doc__
Computes the mean of squares of errors between labels and predictions.

Formula:

```python
loss = mean(square(y_true - y_pred))
```

Args:
    reduction: Type of reduction to apply to the loss. In almost all cases
        this should be `"sum_over_batch_size"`.
        Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
    name: Optional name for the loss instance.
