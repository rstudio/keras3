Extract a diagonal or construct a diagonal array.

@description

# Examples

```r
x <- k_arange(9) |> k_reshape(c(3, 3))
x
```

```
## tf.Tensor(
## [[0 1 2]
##  [3 4 5]
##  [6 7 8]], shape=(3, 3), dtype=int32)
```

```r
k_diag(x)
```

```
## tf.Tensor([0 4 8], shape=(3), dtype=int32)
```

```r
k_diag(x, k = 1)
```

```
## tf.Tensor([1 5], shape=(2), dtype=int32)
```

```r
k_diag(x, k = -1)
```

```
## tf.Tensor([3 7], shape=(2), dtype=int32)
```

```r
k_diag(k_diag(x))
```

```
## tf.Tensor(
## [[0 0 0]
##  [0 4 0]
##  [0 0 8]], shape=(3, 3), dtype=int32)
```

@returns
The extracted diagonal or constructed diagonal tensor.

@param x Input tensor. If `x` is 2-D, returns the k-th diagonal of `x`.
    If `x` is 1-D, return a 2-D tensor with `x` on the k-th diagonal.
@param k The diagonal to consider. Defaults to `0`. Use `k > 0` for diagonals
    above the main diagonal, and `k < 0` for diagonals below
    the main diagonal.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/numpy#diag-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/diag>
