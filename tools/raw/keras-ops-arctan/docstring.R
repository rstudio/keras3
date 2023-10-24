Trigonometric inverse tangent, element-wise.

Args:
    x: Input tensor.

Returns:
    Tensor of the inverse tangent of each element in `x`, in the interval
    `[-pi/2, pi/2]`.

Example:
>>> x = keras.ops.convert_to_tensor([0, 1])
>>> keras.ops.arctan(x)
array([0., 0.7853982], dtype=float32)
