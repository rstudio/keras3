Stops gradient computation.

# Returns
The variable with gradient computation disabled.

# Examples
```python
var = keras.backend.convert_to_tensor(
    [1., 2., 3.],
    dtype="float32"
)
var = keras.ops.stop_gradient(var)
```

@param variable A tensor variable for which the gradient
computation is to be disabled.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/stop_gradient>
