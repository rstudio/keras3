Logarithm of the sigmoid activation function.

It is defined as `f(x) = log(1 / (1 + exp(-x)))`.

Args:
    x: Input tensor.

Returns:
    A tensor with the same shape as `x`.

Example:

>>> x = keras.ops.convert_to_tensor([-0.541391, 0.0, 0.50, 5.0])
>>> keras.ops.log_sigmoid(x)
array([-1.0000418, -0.6931472, -0.474077, -0.00671535], dtype=float32)
