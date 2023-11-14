Hard sigmoid activation function.

@description
It is defined as:

`0 if x < -2.5`, `1 if x > 2.5`, `(0.2 * x) + 0.5 if -2.5 <= x <= 2.5`.

# Examples

```r
x <- k_array(c(-1., 0., 1.))
k_hard_sigmoid(x)
```

```
## tf.Tensor([0.3333333 0.5       0.6666667], shape=(3), dtype=float32)
```

```r
x <- as.array(seq(-5, 5, .1))
plot(x, k_hard_sigmoid(x),
     type = 'l', panel.first = grid(), frame.plot = FALSE)
```

![plot of chunk unnamed-chunk-1](k_hard_sigmoid-unnamed-chunk-1-1.svg)

@returns
A tensor with the same shape as `x`.

@param x
Input tensor.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/nn#hardsigmoid-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/hard_sigmoid>
