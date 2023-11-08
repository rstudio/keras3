keras.ops.arccosh
__signature__
(x)
__doc__
Inverse hyperbolic cosine, element-wise.

Arguments:
    x: Input tensor.

Returns:
    Output tensor of same shape as x.

Example:
>>> x = keras.ops.convert_to_tensor([10, 100])
>>> keras.ops.arccosh(x)
array([2.993223, 5.298292], dtype=float32)
