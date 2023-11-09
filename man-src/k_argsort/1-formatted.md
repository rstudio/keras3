Returns the indices that would sort a tensor.

@description

# Examples
One dimensional array:
```python
x = keras.ops.array([3, 1, 2])
keras.ops.argsort(x)
# array([1, 2, 0], dtype=int32)
```

Two-dimensional array:
```python
x = keras.ops.array([[0, 3], [3, 2], [4, 5]])
x
# array([[0, 3],
#        [3, 2],
#        [4, 5]], dtype=int32)
keras.ops.argsort(x, axis=0)
# array([[0, 1],
#        [1, 0],
#        [2, 2]], dtype=int32)
keras.ops.argsort(x, axis=1)
# array([[0, 1],
#        [1, 0],
#        [0, 1]], dtype=int32)
```

@returns
Tensor of indices that sort `x` along the specified `axis`.

@param x Input tensor.
@param axis Axis along which to sort. Defaults to`-1` (the last axis). If
    `None`, the flattened tensor is used.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/numpy#argsort-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/argsort>
