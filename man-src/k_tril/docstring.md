Return lower triangle of a tensor.

For tensors with `ndim` exceeding 2, `tril` will apply to the
final two axes.

Args:
    x: Input tensor.
    k: Diagonal above which to zero elements. Defaults to `0`. the
        main diagonal. `k < 0` is below it, and `k > 0` is above it.

Returns:
    Lower triangle of `x`, of same shape and data type as `x`.
