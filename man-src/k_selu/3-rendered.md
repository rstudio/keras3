Scaled Exponential Linear Unit (SELU) activation function.

@description
It is defined as:

`f(x) =  scale * alpha * (exp(x) - 1.) for x < 0`,
`f(x) = scale * x for x >= 0`.

# Examples

```r
x <- k_array(c(-1, 0, 1))
k_selu(x)
```

```
## tf.Tensor([-1.11133074  0.          1.05070099], shape=(3), dtype=float64)
```

@returns
A tensor with the same shape as `x`.

@param x Input tensor.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/nn#selu-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/selu>
