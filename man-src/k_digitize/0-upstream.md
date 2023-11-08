keras.ops.digitize
__signature__
(x, bins)
__doc__
Returns the indices of the bins to which each value in `x` belongs.

Args:
    x: Input array to be binned.
    bins: Array of bins. It has to be one-dimensional and monotonically
        increasing.

Returns:
    Output array of indices, of same shape as `x`.

Example:
>>> x = np.array([0.0, 1.0, 3.0, 1.6])
>>> bins = np.array([0.0, 3.0, 4.5, 7.0])
>>> keras.ops.digitize(x, bins)
array([1, 1, 2, 1])
