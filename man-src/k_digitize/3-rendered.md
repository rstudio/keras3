Returns the indices of the bins to which each value in `x` belongs.

@description

# Examples

```r
x <- k_array(c(0.0, 1.0, 3.0, 1.6))
bins <- array(c(0.0, 3.0, 4.5, 7.0))
k_digitize(x, bins)
```

```
## tf.Tensor([1 1 2 1], shape=(4), dtype=int32)
```

```r
# array([1, 1, 2, 1])
```

@returns
Output array of indices, of same shape as `x`.

@param x Input array to be binned.
@param bins Array of bins. It has to be one-dimensional and monotonically
    increasing.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/numpy#digitize-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/digitize>
