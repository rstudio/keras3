Return a slice of an input tensor.

@description
At a high level, this operation is an explicit replacement for array slicing
e.g. `inputs[start_indices:(start_indices + shape)]`.
Unlike slicing via brackets, this operation will accept tensor start
indices on all backends, which is useful when indices dynamically computed
via other tensor operations.


```r
(inputs <- k_arange(5*5) |> k_reshape(c(5, 5)))
```

```
## tf.Tensor(
## [[ 0.  1.  2.  3.  4.]
##  [ 5.  6.  7.  8.  9.]
##  [10. 11. 12. 13. 14.]
##  [15. 16. 17. 18. 19.]
##  [20. 21. 22. 23. 24.]], shape=(5, 5), dtype=float64)
```

```r
start_indices <- c(3, 3)
shape <- c(2, 2)
k_slice(inputs, start_indices, shape)
```

```
## tf.Tensor(
## [[12. 13.]
##  [17. 18.]], shape=(2, 2), dtype=float64)
```

@returns
A tensor, has the same shape and dtype as `inputs`.

@param inputs A tensor, the tensor to be sliced.
@param start_indices A list of length `inputs$ndim`, specifying
    the starting indices for updating.
@param shape The full shape of the returned slice.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/core#slice-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/slice>

