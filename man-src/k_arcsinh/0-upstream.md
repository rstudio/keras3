keras.ops.arcsinh
__signature__
(x)
__doc__
Inverse hyperbolic sine, element-wise.

Arguments:
    x: Input tensor.

Returns:
    Output tensor of same shape as `x`.

Example:
>>> x = keras.ops.convert_to_tensor([1, -1, 0])
>>> keras.ops.arcsinh(x)
array([0.88137364, -0.88137364, 0.0], dtype=float32)
