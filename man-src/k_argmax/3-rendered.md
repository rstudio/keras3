Returns the indices of the maximum values along an axis.

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
k_argmax(x)
```

```
## tf.Tensor(5, shape=(), dtype=int32)
```

```r
k_argmax(x, axis = 1)
```

```
## tf.Tensor([1 1 1], shape=(3), dtype=int32)
```

```r
k_argmax(x, axis = 2)
```

```
## tf.Tensor([2 2], shape=(2), dtype=int32)
```

@returns
Tensor of indices. It has the same shape as `x`, with the dimension
along `axis` removed.

@param x
Input tensor.

@param axis
By default, the index is into the flattened tensor, otherwise
along the specified axis.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/numpy#argmax-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/argmax>
