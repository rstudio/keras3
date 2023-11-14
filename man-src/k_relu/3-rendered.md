Rectified linear unit activation function.

@description
It is defined as `f(x) = max(0, x)`.

# Examples

```r
x1 <- k_convert_to_tensor(c(-1, 0, 1, 0.2))
k_relu(x1)
```

```
## tf.Tensor([0.  0.  1.  0.2], shape=(4), dtype=float32)
```


```r
x <- seq(-10, 10, .1)
plot(x, k_relu(x))
```

![plot of chunk unnamed-chunk-2](k_relu-unnamed-chunk-2-1.svg)
@returns
A tensor with the same shape as `x`.

@param x
Input tensor.

@export
@family nn ops
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/nn#relu-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/relu>

