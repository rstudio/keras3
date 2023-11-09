Returns the indices of the minimum values along an axis.

@description

# Examples

```r
x <- k_arange(6L) |> k_reshape(c(2, 3)) |> k_add(10)
x
```

```
## tf.Tensor(
## [[10. 11. 12.]
##  [13. 14. 15.]], shape=(2, 3), dtype=float32)
```

```r
k_argmin(x)
```

```
## tf.Tensor(0, shape=(), dtype=int32)
```

```r
k_argmin(x, axis = 1)
```

```
## tf.Tensor([0 0 0], shape=(3), dtype=int32)
```

```r
k_argmin(x, axis = 2)
```

```
## tf.Tensor([0 0], shape=(2), dtype=int32)
```

@returns
Tensor of indices. It has the same shape as `x`, with the dimension
along `axis` removed.

@param x Input tensor.
@param axis By default, the index is into the flattened tensor, otherwise
    along the specified axis.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/argmin>
