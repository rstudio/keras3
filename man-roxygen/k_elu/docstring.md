Exponential Linear Unit activation function.

It is defined as:

`f(x) =  alpha * (exp(x) - 1.) for x < 0`, `f(x) = x for x >= 0`.

Args:
    x: Input tensor.
    alpha: A scalar, slope of positive section. Defaults to `1.0`.

Returns:
    A tensor with the same shape as `x`.

Example:

>>> x = np.array([-1., 0., 1.])
>>> x_elu = keras.ops.elu(x)
>>> print(x_elu)
array([-0.63212055, 0., 1.], shape=(3,), dtype=float64)
