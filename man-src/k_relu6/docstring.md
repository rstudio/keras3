Rectified linear unit activation function with upper bound of 6.

It is defined as `f(x) = np.clip(x, 0, 6)`.

Args:
    x: Input tensor.

Returns:
    A tensor with the same shape as `x`.

Example:

>>> x = keras.ops.convert_to_tensor([-3.0, -2.0, 0.1, 0.2, 6.0, 8.0])
>>> keras.ops.relu6(x)
array([0.0, 0.0, 0.1, 0.2, 6.0, 6.0], dtype=float32)
