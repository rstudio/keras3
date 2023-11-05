Computes the error function of `x`, element-wise.

Args:
    x: Input tensor.

Returns:
    A tensor with the same dtype as `x`.

Example:

>>> x = np.array([-3.0, -2.0, -1.0, 0.0, 1.0])
>>> keras.ops.erf(x)
array([-0.99998 , -0.99532, -0.842701,  0.,  0.842701], dtype=float32)
