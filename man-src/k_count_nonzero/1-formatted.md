Counts the number of non-zero values in `x` along the given `axis`.

@description
If no axis is specified then all non-zeros in the tensor are counted.

# Examples
```python
x = keras.ops.array([[0, 1, 7, 0], [3, 0, 2, 19]])
keras.ops.count_nonzero(x)
# 5
keras.ops.count_nonzero(x, axis=0)
# array([1, 1, 2, 1], dtype=int64)
keras.ops.count_nonzero(x, axis=1)
# array([2, 3], dtype=int64)
```

@returns
int or tensor of ints.

@param x Input tensor.
@param axis Axis or tuple of axes along which to count the number of
    non-zeros. Defaults to `None`.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/numpy#countnonzero-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/count_nonzero>
