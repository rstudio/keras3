Exponential Linear Unit activation function.

@description
It is defined as:

`f(x) =  alpha * (exp(x) - 1.) for x < 0`, `f(x) = x for x >= 0`.

# Examples

```r
x <- k_array(c(-1., 0., 1.))
k_elu(x)
```

```
## tf.Tensor([-0.63212056  0.          1.        ], shape=(3), dtype=float64)
```

@returns
A tensor with the same shape as `x`.

@param x Input tensor.
@param alpha A scalar, slope of positive section. Defaults to `1.0`.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/nn#elu-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/elu>
