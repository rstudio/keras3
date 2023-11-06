Computes the logarithm of sum of exponentials of elements in a tensor.

@description

# Examples
```python
x = keras.ops.convert_to_tensor([1., 2., 3.])
logsumexp(x)
# 3.407606
```

@returns
A tensor containing the logarithm of the sum of exponentials of
elements in `x`.

@param x Input tensor.
@param axis An integer or a tuple of integers specifying the axis/axes
    along which to compute the sum. If `None`, the sum is computed
    over all elements. Defaults to`None`.
@param keepdims A boolean indicating whether to keep the dimensions of
    the input tensor when computing the sum. Defaults to`False`.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/logsumexp>
