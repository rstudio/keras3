keras.ops.convert_to_tensor
__signature__
(x, dtype=None)
__doc__
Convert a NumPy array to a tensor.

Args:
    x: A NumPy array.
    dtype: The target type.

Returns:
    A tensor of the specified `dtype`.

Example:

>>> x = np.array([1, 2, 3])
>>> y = keras.ops.convert_to_tensor(x)
