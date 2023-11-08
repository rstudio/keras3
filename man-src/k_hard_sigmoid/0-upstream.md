keras.ops.hard_sigmoid
__signature__
(x)
__doc__
Hard sigmoid activation function.

It is defined as:

`0 if x < -2.5`, `1 if x > 2.5`, `(0.2 * x) + 0.5 if -2.5 <= x <= 2.5`.

Args:
    x: Input tensor.

Returns:
    A tensor with the same shape as `x`.

Example:

>>> x = np.array([-1., 0., 1.])
>>> x_hard_sigmoid = keras.ops.hard_sigmoid(x)
>>> print(x_hard_sigmoid)
array([0.3, 0.5, 0.7], shape=(3,), dtype=float64)
