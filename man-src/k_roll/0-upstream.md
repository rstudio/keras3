keras.ops.roll
__signature__
(x, shift, axis=None)
__doc__
Roll tensor elements along a given axis.

Elements that roll beyond the last position are re-introduced at the first.

Args:
    x: Input tensor.
    shift: The number of places by which elements are shifted.
    axis: The axis along which elements are shifted. By default, the
        array is flattened before shifting, after which the original
        shape is restored.

Returns:
    Output tensor.
