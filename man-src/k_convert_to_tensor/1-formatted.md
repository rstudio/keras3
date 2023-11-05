Convert a NumPy array to a tensor.

# Returns
A tensor of the specified `dtype`.

# Examples
```python
x = np.array([1, 2, 3])
y = keras.ops.convert_to_tensor(x)
```

@param x A NumPy array.
@param dtype The target type.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/convert_to_tensor>
