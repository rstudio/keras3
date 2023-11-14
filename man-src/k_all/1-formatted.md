Test whether all array elements along a given axis evaluate to `True`.

@description

# Examples
```python
x = keras.ops.convert_to_tensor([True, False])
keras.ops.all(x)
# array(False, shape=(), dtype=bool)
```

```python
x = keras.ops.convert_to_tensor([[True, False], [True, True]])
keras.ops.all(x, axis=0)
# array([ True False], shape=(2,), dtype=bool)
```

`keepdims=True` outputs a tensor with dimensions reduced to one.
```python
x = keras.ops.convert_to_tensor([[True, False], [True, True]])
keras.ops.all(x, keepdims=True)
# array([[False]], shape=(1, 1), dtype=bool)
```

@returns
The tensor containing the logical AND reduction over the `axis`.

@param x
Input tensor.

@param axis
An integer or tuple of integers that represent the axis along
which a logical AND reduction is performed. The default
(`axis=None`) is to perform a logical AND over all the dimensions
of the input array. `axis` may be negative, in which case it counts
for the last to the first axis.

@param keepdims
If `True`, axes which are reduced are left in the result as
dimensions with size one. With this option, the result will
broadcast correctly against the input array. Defaults to`False`.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/numpy#all-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/all>
