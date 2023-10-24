Compute the arithmetic mean along the specified axes.

Args:
    x: Input tensor.
    axis: Axis or axes along which the means are computed. The default
        is to compute the mean of the flattened tensor.
    keepdims: If this is set to `True`, the axes which are reduced are left
        in the result as dimensions with size one.

Returns:
    Output tensor containing the mean values.
