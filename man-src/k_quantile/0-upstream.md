keras.ops.quantile
__signature__
(
  x,
  q,
  axis=None,
  method='linear',
  keepdims=False
)
__doc__
Compute the q-th quantile(s) of the data along the specified axis.

Args:
    x: Input tensor.
    q: Probability or sequence of probabilities for the quantiles to
        compute. Values must be between 0 and 1 inclusive.
    axis: Axis or axes along which the quantiles are computed. Defaults to
        `axis=None` which is to compute the quantile(s) along a flattened
        version of the array.
    method: A string specifies the method to use for estimating the
        quantile. Available methods are `"linear"`, `"lower"`, `"higher"`,
        `"midpoint"`, and `"nearest"`. Defaults to `"linear"`.
        If the desired quantile lies between two data points `i < j`:
        - `"linear"`: `i + (j - i) * fraction`, where fraction is the
            fractional part of the index surrounded by `i` and `j`.
        - `"lower"`: `i`.
        - `"higher"`: `j`.
        - `"midpoint"`: `(i + j) / 2`
        - `"nearest"`: `i` or `j`, whichever is nearest.
    keepdims: If this is set to `True`, the axes which are reduce
        are left in the result as dimensions with size one.

Returns:
    The quantile(s). If `q` is a single probability and `axis=None`, then
    the result is a scalar. If multiple probabilies levels are given, first
    axis of the result corresponds to the quantiles. The other axes are the
    axes that remain after the reduction of `x`.
