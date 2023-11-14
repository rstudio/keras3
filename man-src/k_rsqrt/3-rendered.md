Computes reciprocal of square root of x element-wise.

@description

# Examples

```r
x <- k_convert_to_tensor(c(1, 10, 100))
k_rsqrt(x)
```

```
## tf.Tensor([1.         0.31622776 0.1       ], shape=(3), dtype=float32)
```

```r
# array([1, 0.31622776, 0.1], dtype=float32)
```

@returns
A tensor with the same dtype as `x`.

@param x
input tensor

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/core#rsqrt-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/rsqrt>

