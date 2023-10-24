Computes the mean absolute percentage error between `y_true` & `y_pred`.

Formula:

```python
loss = 100 * mean(abs((y_true - y_pred) / y_true))
```

Args:
    reduction: Type of reduction to apply to the loss. In almost all cases
        this should be `"sum_over_batch_size"`.
        Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
    name: Optional name for the loss instance.
