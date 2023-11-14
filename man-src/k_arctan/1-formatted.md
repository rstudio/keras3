Trigonometric inverse tangent, element-wise.

@description

# Examples
```python
x = keras.ops.convert_to_tensor([0, 1])
keras.ops.arctan(x)
# array([0., 0.7853982], dtype=float32)
```

@returns
Tensor of the inverse tangent of each element in `x`, in the interval
`[-pi/2, pi/2]`.

@param x
Input tensor.

@export
@family numpy ops
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/numpy#arctan-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/arctan>
