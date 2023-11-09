Inverse sine, element-wise.

@description

# Examples

```r
x <- k_convert_to_tensor(c(1, -1, 0))
k_arcsin(x)
```

```
## tf.Tensor([ 1.5707963 -1.5707963  0.       ], shape=(3), dtype=float32)
```

@returns
Tensor of the inverse sine of each element in `x`, in radians and in
the closed interval `[-pi/2, pi/2]`.

@param x Input tensor.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/numpy#arcsin-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/arcsin>
