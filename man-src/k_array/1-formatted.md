Create a tensor.

@description

# Examples
```python
keras.ops.array([1, 2, 3])
# array([1, 2, 3], dtype=int32)
```

```python
keras.ops.array([1, 2, 3], dtype="float32")
# array([1., 2., 3.], dtype=float32)
```

@returns
A tensor.

@param x
Input tensor.

@param dtype
The desired data-type for the tensor.

@export
@family numpy ops
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/numpy#array-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/array>
