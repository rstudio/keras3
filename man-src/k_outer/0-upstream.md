keras.ops.outer
__signature__
(x1, x2)
__doc__
Compute the outer product of two vectors.

Given two vectors `x1` and `x2`, the outer product is:

```
out[i, j] = x1[i] * x2[j]
```

Args:
    x1: First input tensor.
    x2: Second input tensor.

Returns:
    Outer product of `x1` and `x2`.
