Log-softmax activation function.

@description
It is defined as:
`f(x) = x - max(x) - log(sum(exp(x - max(x))))`

# Examples

```r
x <- k_array(c(-1., 0., 1.))
k_log_softmax(x)
```

```
## tf.Tensor([-2.407606   -1.4076059  -0.40760595], shape=(3), dtype=float32)
```

@returns
A tensor with the same shape as `x`.

@param x
Input tensor.

@param axis
Integer, axis along which the log-softmax is applied.
Defaults to `-1`.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/nn#logsoftmax-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/log_softmax>

