Return specified diagonals.

@description
If `x` is 2-D, returns the diagonal of `x` with the given offset, i.e., the
collection of elements of the form `x[i, i+offset]`.

If `x` has more than two dimensions, the axes specified by `axis1`
and `axis2` are used to determine the 2-D sub-array whose diagonal
is returned.

The shape of the resulting array can be determined by removing `axis1`
and `axis2` and appending an index to the right equal to the size of
the resulting diagonals.

# Examples

```r
x <- k_arange(4L) |> k_reshape(c(2, 2))
x
```

```
## tf.Tensor(
## [[0 1]
##  [2 3]], shape=(2, 2), dtype=int32)
```

```r
k_diagonal(x)
```

```
## tf.Tensor([0 3], shape=(2), dtype=int32)
```

```r
k_diagonal(x, offset = 1)
```

```
## tf.Tensor([1], shape=(1), dtype=int32)
```

```r
x <- k_array(1:8) |> k_reshape(c(2, 2, 2))
x
```

```
## tf.Tensor(
## [[[1 2]
##   [3 4]]
##
##  [[5 6]
##   [7 8]]], shape=(2, 2, 2), dtype=int64)
```

```r
x |> k_diagonal(0)
```

```
## tf.Tensor(
## [[1 7]
##  [2 8]], shape=(2, 2), dtype=int64)
```

```r
x |> k_diagonal(0, 1, 2) # same as above, the default
```

```
## tf.Tensor(
## [[1 7]
##  [2 8]], shape=(2, 2), dtype=int64)
```

```r
x |> k_diagonal(0, 2, 3)
```

```
## tf.Tensor(
## [[1 4]
##  [5 8]], shape=(2, 2), dtype=int64)
```

@returns
Tensor of diagonals.

@param x Input tensor.
@param offset Offset of the diagonal from the main diagonal.
    Can be positive or negative. Defaults to `0`.(main diagonal).
@param axis1 Axis to be used as the first axis of the 2-D sub-arrays.
    Defaults to `1`.(first axis).
@param axis2 Axis to be used as the second axis of the 2-D sub-arrays.
    Defaults to `2` (second axis).

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/numpy#diagonal-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/diagonal>
