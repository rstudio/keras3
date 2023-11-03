Gaussian Error Linear Unit (GELU) activation function.

If `approximate` is `True`, it is defined as:
`f(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))`

Or if `approximate` is `False`, it is defined as:
`f(x) = x * P(X <= x) = 0.5 * x * (1 + erf(x / sqrt(2)))`,
where `P(X) ~ N(0, 1)`.

Args:
    x: Input tensor.
    approximate: Approximate version of GELU activation. Defaults to `True`.

Returns:
    A tensor with the same shape as `x`.

Example:

>>> x = np.array([-1., 0., 1.])
>>> x_gelu = keras.ops.gelu(x)
>>> print(x_gelu)
array([-0.15865525, 0., 0.84134475], shape=(3,), dtype=float64)
