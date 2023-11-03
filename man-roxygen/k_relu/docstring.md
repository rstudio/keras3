Rectified linear unit activation function.

It is defined as `f(x) = max(0, x)`.

Args:
    x: Input tensor.

Returns:
    A tensor with the same shape as `x`.

Example:

>>> x1 = keras.ops.convert_to_tensor([-1.0, 0.0, 1.0, 0.2])
>>> keras.ops.relu(x1)
array([0.0, 0.0, 1.0, 0.2], dtype=float32)
