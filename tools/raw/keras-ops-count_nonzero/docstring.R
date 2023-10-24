Counts the number of non-zero values in `x` along the given `axis`.

If no axis is specified then all non-zeros in the tensor are counted.

Args:
    x: Input tensor.
    axis: Axis or tuple of axes along which to count the number of
        non-zeros. Defaults to `None`.

Returns:
    int or tensor of ints.

Examples:
>>> x = keras.ops.array([[0, 1, 7, 0], [3, 0, 2, 19]])
>>> keras.ops.count_nonzero(x)
5
>>> keras.ops.count_nonzero(x, axis=0)
array([1, 1, 2, 1], dtype=int64)
>>> keras.ops.count_nonzero(x, axis=1)
array([2, 3], dtype=int64)
