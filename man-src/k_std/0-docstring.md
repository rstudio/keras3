Compute the standard deviation along the specified axis.

Args:
    x: Input tensor.
    axis: Axis along which to compute standard deviation.
        Default is to compute the standard deviation of the
        flattened tensor.
    keepdims: If this is set to `True`, the axes which are reduced are left
        in the result as dimensions with size one.

Returns:
    Output tensor containing the standard deviation values.
