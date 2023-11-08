keras.ops.softplus
__signature__
(x)
__doc__
Softplus activation function.

It is defined as `f(x) = log(exp(x) + 1)`, where `log` is the natural
logarithm and `exp` is the exponential function.

Args:
    x: Input tensor.

Returns:
    A tensor with the same shape as `x`.

Example:

>>> x = keras.ops.convert_to_tensor([-0.555, 0.0, 0.555])
>>> keras.ops.softplus(x)
array([0.45366603, 0.6931472, 1.008666], dtype=float32)
