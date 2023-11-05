Create a tensor.

# Returns
A tensor.

# Examples
```python
keras.ops.array([1, 2, 3])
# array([1, 2, 3], dtype=int32)
```

```python
keras.ops.array([1, 2, 3], dtype="float32")
# array([1., 2., 3.], dtype=float32)
```

@param x Input tensor.
@param dtype The desired data-type for the tensor.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/array>
