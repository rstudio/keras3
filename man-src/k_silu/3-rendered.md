Sigmoid Linear Unit (SiLU) activation function, also known as Swish.

@description
The SiLU activation function is computed by the sigmoid function multiplied
by its input. It is defined as `f(x) = x * sigmoid(x)`.

# Examples

```r
x <- k_convert_to_tensor(c(-6, 1, 0, 1, 6))
k_sigmoid(x)
```

```
## tf.Tensor([0.00247262 0.7310586  0.5        0.7310586  0.99752736], shape=(5), dtype=float32)
```

```r
k_silu(x)
```

```
## tf.Tensor([-0.01483574  0.7310586   0.          0.7310586   5.985164  ], shape=(5), dtype=float32)
```

@returns
A tensor with the same shape as `x`.

@param x Input tensor.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/nn#silu-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/silu>
