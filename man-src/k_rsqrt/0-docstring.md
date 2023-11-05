Computes reciprocal of square root of x element-wise.

Args:
    x: input tensor

Returns:
    A tensor with the same dtype as `x`.

Example:

>>> x = keras.ops.convert_to_tensor([1.0, 10.0, 100.0])
>>> keras.ops.rsqrt(x)
array([1.0, 0.31622776, 0.1], dtype=float32)
