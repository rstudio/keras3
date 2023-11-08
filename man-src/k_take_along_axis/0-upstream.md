keras.ops.take_along_axis
__signature__
(x, indices, axis=None)
__doc__
Select values from `x` at the 1-D `indices` along the given axis.

Args:
    x: Source tensor.
    indices: The indices of the values to extract.
    axis: The axis over which to select values. By default, the flattened
        input tensor is used.

Returns:
    The corresponding tensor of values.
