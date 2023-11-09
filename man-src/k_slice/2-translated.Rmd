Return a slice of an input tensor.

@description
At a high level, this operation is an explicit replacement for array slicing
e.g. `inputs[start_indices: start_indices + shape]`.
Unlike slicing via brackets, this operation will accept tensor start
indices on all backends, which is useful when indices dynamically computed
via other tensor operations.

```python
inputs = np.zeros((5, 5))
start_indices = np.array([3, 3])
shape = np.array([2, 2])
inputs = keras.ops.slice(inputs, start_indices, updates)
```

@returns
    A tensor, has the same shape and dtype as `inputs`.

@param inputs A tensor, the tensor to be updated.
@param start_indices A list/tuple of shape `(inputs.ndim,)`, specifying
    the starting indices for updating.
@param shape The full shape of the returned slice.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/core#slice-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/slice>
