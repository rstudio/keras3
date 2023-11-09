keras.ops.convert_to_tensor
__signature__
(x, dtype=None, sparse=None)
__doc__
Convert a NumPy array to a tensor.

Args:
    x: A NumPy array.
    dtype: The target type.
    sparse: Whether to keep sparse tensors. `False` will cause sparse
        tensors to be densified. The default value of `None` means that
        sparse tensors are kept only if the backend supports them.

Returns:
    A tensor of the specified `dtype`.

Example:

>>> x = np.array([1, 2, 3])
>>> y = keras.ops.convert_to_tensor(x)
