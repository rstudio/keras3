Logarithm of the sigmoid activation function.

@description
It is defined as `f(x) = log(1 / (1 + exp(-x)))`.

# Examples

```r
x <- k_convert_to_tensor(c(-0.541391, 0.0, 0.50, 5.0))
k_log_sigmoid(x)
```

```
## tf.Tensor([-1.0000418  -0.6931472  -0.474077   -0.00671535], shape=(4), dtype=float32)
```

@returns
A tensor with the same shape as `x`.

@param x
Input tensor.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/nn#logsigmoid-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/log_sigmoid>

