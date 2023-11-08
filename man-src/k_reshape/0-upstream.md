keras.ops.reshape
__signature__
(x, new_shape)
__doc__
Gives a new shape to a tensor without changing its data.

Args:
    x: Input tensor.
    new_shape: The new shape should be compatible with the original shape.
        One shape dimension can be -1 in which case the value is
        inferred from the length of the array and remaining dimensions.

Returns:
    The reshaped tensor.
