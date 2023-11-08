keras.ops.median
__signature__
(x, axis=None, keepdims=False)
__doc__
Compute the median along the specified axis.

Args:
    x: Input tensor.
    axis: Axis or axes along which the medians are computed. Defaults to
        `axis=None` which is to compute the median(s) along a flattened
        version of the array.
    keepdims: If this is set to `True`, the axes which are reduce
        are left in the result as dimensions with size one.

Returns:
    The output tensor.
