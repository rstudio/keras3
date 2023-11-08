keras.ops.amax
__signature__
(x, axis=None, keepdims=False)
__doc__
Returns the maximum of an array or maximum value along an axis.

Args:
    x: Input tensor.
    axis: Axis along which to compute the maximum.
        By default (`axis=None`), find the maximum value in all the
        dimensions of the input array.
    keepdims: If `True`, axes which are reduced are left in the result as
        dimensions that are broadcast to the size of the original
        input tensor. Defaults to `False`.

Returns:
    An array with the maximum value. If `axis=None`, the result is a scalar
    value representing the maximum element in the entire array. If `axis` is
    given, the result is an array with the maximum values along
    the specified axis.

Examples:
>>> x = keras.ops.convert_to_tensor([[1, 3, 5], [2, 3, 6]])
>>> keras.ops.amax(x)
array(6, dtype=int32)

>>> x = keras.ops.convert_to_tensor([[1, 6, 8], [1, 5, 2]])
>>> keras.ops.amax(x, axis=0)
array([1, 6, 8], dtype=int32)

>>> x = keras.ops.convert_to_tensor([[1, 6, 8], [1, 5, 2]])
>>> keras.ops.amax(x, axis=1, keepdims=True)
array([[8], [5]], dtype=int32)
