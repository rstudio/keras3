Returns the minimum of an array or minimum value along an axis.

@description

# Returns
An array with the minimum value. If `axis=None`, the result is a scalar
value representing the minimum element in the entire array. If `axis` is
given, the result is an array with the minimum values along
the specified axis.

# Examples
```python
x = keras.ops.convert_to_tensor([1, 3, 5, 2, 3, 6])
keras.ops.amin(x)
# array(1, dtype=int32)
```

```python
x = keras.ops.convert_to_tensor([[1, 6, 8], [7, 5, 3]])
keras.ops.amin(x, axis=0)
# array([1,5,3], dtype=int32)
```

```python
x = keras.ops.convert_to_tensor([[1, 6, 8], [7, 5, 3]])
keras.ops.amin(x, axis=1, keepdims=True)
# array([[1],[3]], dtype=int32)
```

@param x Input tensor.
@param axis Axis along which to compute the minimum.
    By default (`axis=None`), find the minimum value in all the
    dimensions of the input array.
@param keepdims If `True`, axes which are reduced are left in the result as
    dimensions that are broadcast to the size of the original
    input tensor. Defaults to `False`.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/amin>
