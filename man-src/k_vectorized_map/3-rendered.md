Parallel map of `function` on axis 0 of tensor(s) `elements`.

@description
Schematically, `vectorized_map` implements the following,
in the case of a single tensor input `elements`:


```r
k_vectorized_map <- function(elements, f) {
  apply(elements, 1, f)
}
```

In the case of an iterable of tensors `elements`,
it implements the following:


```r
k_vectorized_map <- function(elements, f) {
    batch_size <- elements[[1]]$shape[[1]]
    outputs <- vector("list", batch_size)
    outputs <- lapply(seq(batch_size), \(index) {
        f(lapply(elements, \(e) e[index, all_dims()]))
    }
    k_stack(outputs)
}
```

In this case, `function` is expected to take as input
a single list of tensor arguments.



```r
(x <- k_arange(4*4) |> k_reshape(c(4,4)))
```

```
## tf.Tensor(
## [[ 0.  1.  2.  3.]
##  [ 4.  5.  6.  7.]
##  [ 8.  9. 10. 11.]
##  [12. 13. 14. 15.]], shape=(4, 4), dtype=float64)
```

```r
x |> k_vectorized_map(\(row) {row + 10})
```

```
## tf.Tensor(
## [[10. 11. 12. 13.]
##  [14. 15. 16. 17.]
##  [18. 19. 20. 21.]
##  [22. 23. 24. 25.]], shape=(4, 4), dtype=float64)
```

```r
list(x, x, x) |> k_vectorized_map(\(rows) Reduce(`+`, rows))
```

```
## tf.Tensor(
## [[ 0.  3.  6.  9.]
##  [12. 15. 18. 21.]
##  [24. 27. 30. 33.]
##  [36. 39. 42. 45.]], shape=(4, 4), dtype=float64)
```

@param elements
see description

@param f
A function taking either a tensor, or list of tensors.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/vectorized_map>

