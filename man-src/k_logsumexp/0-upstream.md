keras.ops.logsumexp
__signature__
(x, axis=None, keepdims=False)
__doc__
Computes the logarithm of sum of exponentials of elements in a tensor.

Args:
    x: Input tensor.
    axis: An integer or a tuple of integers specifying the axis/axes
        along which to compute the sum. If `None`, the sum is computed
        over all elements. Defaults to`None`.
    keepdims: A boolean indicating whether to keep the dimensions of
        the input tensor when computing the sum. Defaults to`False`.

Returns:
    A tensor containing the logarithm of the sum of exponentials of
    elements in `x`.

Example:

>>> x = keras.ops.convert_to_tensor([1., 2., 3.])
>>> logsumexp(x)
3.407606
