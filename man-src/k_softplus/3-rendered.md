Softplus activation function.

@description
It is defined as `f(x) = log(exp(x) + 1)`, where `log` is the natural
logarithm and `exp` is the exponential function.

# Examples

```r
x <- k_convert_to_tensor(c(-0.555, 0, 0.555))
k_softplus(x)
```

```
## tf.Tensor([0.45366603 0.6931472  1.008666  ], shape=(3), dtype=float32)
```

```r
x <- seq(-10, 10, .1)
plot(x, k_softplus(x))
```

![plot of chunk unnamed-chunk-2](k_softplus-unnamed-chunk-2-1.svg)
@returns
A tensor with the same shape as `x`.

@param x
Input tensor.

@export
@family nn ops
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/nn#softplus-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/softplus>

