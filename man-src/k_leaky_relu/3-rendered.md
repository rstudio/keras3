Leaky version of a Rectified Linear Unit activation function.

@description
It allows a small gradient when the unit is not active, it is defined as:

`f(x) = alpha * x for x < 0` or `f(x) = x for x >= 0`.

# Examples

```r
x <- k_array(c(-1., 0., 1.))
k_leaky_relu(x)
```

```
## tf.Tensor([-0.2  0.   1. ], shape=(3), dtype=float32)
```

```r
# array([-0.2,  0. ,  1. ], shape=(3,), dtype=float64)
```

```r
x <- seq(-5, 5, .1)
plot(x, k_leaky_relu(x),
     type = 'l', panel.first = grid())
```

![plot of chunk unnamed-chunk-2](k_leaky_relu-unnamed-chunk-2-1.svg)

@returns
A tensor with the same shape as `x`.

@param x Input tensor.
@param negative_slope Slope of the activation function at x < 0.
    Defaults to `0.2`.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/nn#leakyrelu-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/leaky_relu>

