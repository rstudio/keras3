Softsign activation function.

@description
It is defined as `f(x) = x / (abs(x) + 1)`.

# Examples

```r
x <- k_convert_to_tensor(c(-0.100, -10.0, 1.0, 0.0, 100.0))
k_softsign(x)
```

```
## tf.Tensor([-0.09090909 -0.90909094  0.5         0.          0.990099  ], shape=(5), dtype=float32)
```

```r
x <- seq(-10, 10, .1)
plot(x, k_softsign(x), ylim = c(-1, 1))
```

![plot of chunk unnamed-chunk-2](k_softsign-unnamed-chunk-2-1.svg)

@returns
A tensor with the same shape as `x`.

@param x
Input tensor.

@export
@family nn ops
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/nn#softsign-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/softsign>

