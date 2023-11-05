Scaled Exponential Linear Unit (SELU) activation function.

It is defined as:

`f(x) =  scale * alpha * (exp(x) - 1.) for x < 0`,
`f(x) = scale * x for x >= 0`.

Args:
    x: Input tensor.

Returns:
    A tensor with the same shape as `x`.

Example:

>>> x = np.array([-1., 0., 1.])
>>> x_selu = keras.ops.selu(x)
>>> print(x_selu)
array([-1.11133055, 0., 1.05070098], shape=(3,), dtype=float64)
