Returns a tensor of shape `shape` where `indices` are set to `values`.

@description
At a high level, this operation does `zeros[indices] = updates` and
returns the output. It is equivalent to:


```r
output <- k_scatter_update(k_zeros(shape), indices, values)
```

# Examples

```r
indices <- rbind(c(1, 2), c(2, 2))
values <- k_array(c(1, 1))
k_scatter(indices, values, shape= c(2, 2))
```

```
## tf.Tensor(
## [[0. 1.]
##  [0. 1.]], shape=(2, 2), dtype=float32)
```

@param indices A tensor or list specifying
    indices for the values in `values`.
@param values A tensor, the values to be set at `indices`.
@param shape Shape of the output tensor.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/core#scatter-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/scatter>
