Inverse hyperbolic sine, element-wise.

@description

# Examples

```r
x <- k_convert_to_tensor(c(1, -1, 0))
k_arcsinh(x)
```

```
## tf.Tensor([ 0.8813736 -0.8813736  0.       ], shape=(3), dtype=float32)
```

@returns
Output tensor of same shape as `x`.

@param x Input tensor.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/numpy#arcsinh-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/arcsinh>
