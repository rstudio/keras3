keras.ops.triu
__signature__
(x, k=0)
__doc__
Return upper triangle of a tensor.

For tensors with `ndim` exceeding 2, `triu` will apply to the
final two axes.

Args:
    x: Input tensor.
    k: Diagonal below which to zero elements. Defaults to `0`. the
        main diagonal. `k < 0` is below it, and `k > 0` is above it.

Returns:
    Upper triangle of `x`, of same shape and data type as `x`.
