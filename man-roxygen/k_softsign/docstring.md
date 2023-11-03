Softsign activation function.

It is defined as `f(x) = x / (abs(x) + 1)`.

Args:
    x: Input tensor.

Returns:
    A tensor with the same shape as `x`.

Example:

>>> x = keras.ops.convert_to_tensor([-0.100, -10.0, 1.0, 0.0, 100.0])
>>> keras.ops.softsign(x)
Array([-0.09090909, -0.90909094, 0.5, 0.0, 0.990099], dtype=float32)
