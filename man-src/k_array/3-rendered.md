Create a tensor.

@description

# Examples

```r
k_array(c(1, 2, 3))
```

```
## tf.Tensor([1. 2. 3.], shape=(3), dtype=float64)
```

```r
k_array(c(1, 2, 3), dtype = "float32")
```

```
## tf.Tensor([1. 2. 3.], shape=(3), dtype=float32)
```

@returns
A tensor.

@param x Input tensor.
@param dtype The desired data-type for the tensor.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/numpy#array-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/array>
