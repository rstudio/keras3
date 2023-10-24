Computes the cosine similarity between `y_true` & `y_pred`.

Note that it is a number between -1 and 1. When it is a negative number
between -1 and 0, 0 indicates orthogonality and values closer to -1
indicate greater similarity. This makes it usable as a loss function in a
setting where you try to maximize the proximity between predictions and
targets. If either `y_true` or `y_pred` is a zero vector, cosine similarity
will be 0 regardless of the proximity between predictions and targets.

Formula:

```python
loss = -sum(l2_norm(y_true) * l2_norm(y_pred))
```

Args:
    axis: The axis along which the cosine similarity is computed
        (the features axis). Defaults to `-1`.
    reduction: Type of reduction to apply to the loss. In almost all cases
        this should be `"sum_over_batch_size"`.
        Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
    name: Optional name for the loss instance.
