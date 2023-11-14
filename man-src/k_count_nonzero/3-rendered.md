Counts the number of non-zero values in `x` along the given `axis`.

@description
If no axis is specified then all non-zeros in the tensor are counted.

# Examples

```r
x <- k_array(rbind(c(0, 1, 7, 0),
                   c(3, 0, 2, 19)))
k_count_nonzero(x)
```

```
## tf.Tensor(5, shape=(), dtype=int64)
```

```r
k_count_nonzero(x, axis = 1)
```

```
## tf.Tensor([1 1 2 1], shape=(4), dtype=int64)
```

```r
k_count_nonzero(x, axis = 2)
```

```
## tf.Tensor([2 3], shape=(2), dtype=int64)
```

@returns
An integer or a tensor of integers.

@param x
Input tensor.

@param axis
Axis or a tuple of axes along which to count the number of
non-zeros. Defaults to `NULL`.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/numpy#countnonzero-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/count_nonzero>
