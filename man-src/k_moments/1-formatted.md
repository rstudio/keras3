Calculates the mean and variance of `x`.

@description
The mean and variance are calculated by aggregating the contents of `x`
across `axes`. If `x` is 1-D and `axes = [0]` this is just the mean and
variance of a vector.

# Examples
```python
x = keras.ops.convert_to_tensor([0, 1, 2, 3, 100], dtype="float32")
keras.ops.moments(x, axes=[0])
# (array(21.2, dtype=float32), array(1553.3601, dtype=float32))
```

@returns
A tuple containing two tensors - mean and variance.

@param x
Input tensor.

@param axes
A list of axes which to compute mean and variance.

@param keepdims
If this is set to `True`, the axes which are reduced are left
in the result as dimensions with size one.

@param synchronized
Only applicable with the TensorFlow backend.
If `True`, synchronizes the global batch statistics (mean and
variance) across all devices at each training step in a
distributed training strategy. If `False`, each replica uses its own
local batch statistics.

@export
@family nn ops
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/moments>
