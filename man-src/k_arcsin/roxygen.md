Inverse sine, element-wise.

@description

# Returns
Tensor of the inverse sine of each element in `x`, in radians and in
the closed interval `[-pi/2, pi/2]`.

# Examples
```python
x = keras.ops.convert_to_tensor([1, -1, 0])
keras.ops.arcsin(x)
# array([ 1.5707964, -1.5707964,  0.], dtype=float32)
```

@param x Input tensor.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/arcsin>
