Returns the indices that would sort a tensor.

@description

# Examples
One dimensional array:

```r
x <- k_array(c(3, 1, 2))
k_argsort(x)
```

```
## tf.Tensor([1 2 0], shape=(3), dtype=int32)
```

Two-dimensional array:

```r
x <- k_array(rbind(c(0, 3),
                   c(3, 2),
                   c(4, 5)), dtype = "int32")
k_argsort(x, axis = 1)
```

```
## tf.Tensor(
## [[0 1]
##  [1 0]
##  [2 2]], shape=(3, 2), dtype=int32)
```

```r
k_argsort(x, axis = 2)
```

```
## tf.Tensor(
## [[0 1]
##  [1 0]
##  [0 1]], shape=(3, 2), dtype=int32)
```

@returns
Tensor of indices that sort `x` along the specified `axis`.

@param x Input tensor.
@param axis Axis along which to sort. Defaults to `-1` (the last axis). If
    `NULL`, the flattened tensor is used.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/numpy#argsort-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/argsort>
