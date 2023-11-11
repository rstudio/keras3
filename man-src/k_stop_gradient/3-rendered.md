Stops gradient computation.

@description

# Examples

```r
var <- k_convert_to_tensor(c(1, 2, 3), dtype="float32")
var <- k_stop_gradient(var)
```

@returns
The variable with gradient computation disabled.

@param variable A tensor variable for which the gradient
computation is to be disabled.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/core#stopgradient-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/stop_gradient>

