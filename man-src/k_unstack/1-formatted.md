Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.

@description

# Examples
```python
x = keras.ops.array([[1, 2], [3, 4]])
keras.ops.unstack(x, axis=0)
# [array([1, 2]), array([3, 4])]
```

@returns
A list of tensors unpacked along the given axis.

@param x The input tensor.
@param num The length of the dimension axis. Automatically inferred
    if `None`.
@param axis The axis along which to unpack.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/unstack>
