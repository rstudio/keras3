Compute the absolute value element-wise.

`keras.ops.abs` is a shorthand for this function.

Args:
    x: Input tensor.

Returns:
    An array containing the absolute value of each element in `x`.

Example:

>>> x = keras.ops.convert_to_tensor([-1.2, 1.2])
>>> keras.ops.absolute(x)
array([1.2, 1.2], dtype=float32)
