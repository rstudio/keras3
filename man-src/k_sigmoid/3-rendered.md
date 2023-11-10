Sigmoid activation function.

@description
It is defined as `f(x) = 1 / (1 + exp(-x))`.

# Examples

```r
x <- k_convert_to_tensor(c(-6, 1, 0, 1, 6))
k_sigmoid(x)
```

```
## tf.Tensor([0.00247262 0.7310586  0.5        0.7310586  0.99752736], shape=(5), dtype=float32)
```

@returns
A tensor with the same shape as `x`.

@param x Input tensor.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/nn#sigmoid-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/sigmoid>
