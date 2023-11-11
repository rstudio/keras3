Solves for `x` in the equation `a * x = b`.

@description

# Examples

```r
a <- k_array(c(1, 2, 4, 5), dtype="float32") |> k_reshape(c(2, 2))
b <- k_array(c(2, 4, 8, 10), dtype="float32") |> k_reshape(c(2, 2))
k_solve(a, b)
```

```
## tf.Tensor(
## [[2. 0.]
##  [0. 2.]], shape=(2, 2), dtype=float32)
```

@returns
A tensor with the same shape and dtype as `a`.

@param a Input tensor.
@param b Input tensor.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/solve>
