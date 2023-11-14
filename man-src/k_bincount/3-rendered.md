Count the number of occurrences of each value in a tensor of integers.

@description

# Examples

```r
(x <- k_array(c(1, 2, 2, 3), dtype = "uint8"))
```

```
## tf.Tensor([1 2 2 3], shape=(4), dtype=uint8)
```

```r
k_bincount(x)
```

```
## tf.Tensor([0 1 2 1], shape=(4), dtype=int32)
```

```r
(weights <- x / 2)
```

```
## tf.Tensor([0.5 1.  1.  1.5], shape=(4), dtype=float32)
```

```r
k_bincount(x, weights = weights)
```

```
## tf.Tensor([0.  0.5 2.  1.5], shape=(4), dtype=float32)
```

```r
minlength <- as.integer(k_max(x) + 1 + 2) # 6
k_bincount(x, minlength = minlength)
```

```
## tf.Tensor([0 1 2 1 0 0], shape=(6), dtype=int32)
```

@returns
1D tensor where each element gives the number of occurrence(s) of its
index value in x. Its length is the maximum between `max(x) + 1` and
minlength.

@param x
Input tensor.
It must be of dimension 1, and it must only contain non-negative
integer(s).

@param weights
Weight tensor.
It must have the same length as `x`. The default value is `NULL`.
If specified, `x` is weighted by it, i.e. if `n = x[i]`,
`out[n] += weight[i]` instead of the default behavior `out[n] += 1`.

@param minlength
An integer.
The default value is 0. If specified, there will be at least
this number of bins in the output tensor. If greater than
`max(x) + 1`, each value of the output at an index higher than
`max(x)` is set to 0.

@export
@family numpy ops
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/numpy#bincount-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/bincount>
