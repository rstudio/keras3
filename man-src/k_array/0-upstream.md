keras.ops.array
__signature__
(x, dtype=None)
__doc__
Create a tensor.

Args:
    x: Input tensor.
    dtype: The desired data-type for the tensor.

Returns:
    A tensor.

Examples:
>>> keras.ops.array([1, 2, 3])
array([1, 2, 3], dtype=int32)

>>> keras.ops.array([1, 2, 3], dtype="float32")
array([1., 2., 3.], dtype=float32)
