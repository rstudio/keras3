Computes the logarithm of the hyperbolic cosine of the prediction error.

Formula:

```python
error = y_pred - y_true
logcosh = mean(log((exp(error) + exp(-error))/2), axis=-1)`
```
where x is the error `y_pred - y_true`.

Args:
    reduction: Type of reduction to apply to loss. Options are `"sum"`,
        `"sum_over_batch_size"` or `None`. Defaults to
        `"sum_over_batch_size"`.
    name: Optional name for the instance.
