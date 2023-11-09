Test whether any array element along a given axis evaluates to `TRUE`.

@description

# Examples

```r
x <- k_array(c(TRUE, FALSE))
k_any(x)
```

```
## tf.Tensor(True, shape=(), dtype=bool)
```


```r
(x <- k_reshape(c(FALSE, FALSE, FALSE,
                  TRUE, FALSE, FALSE), c(2, 3)))
```

```
## tf.Tensor(
## [[False False False]
##  [ True False False]], shape=(2, 3), dtype=bool)
```

```r
k_any(x, axis = 1)
```

```
## tf.Tensor([ True False False], shape=(3), dtype=bool)
```

```r
k_any(x, axis = 2)
```

```
## tf.Tensor([False  True], shape=(2), dtype=bool)
```

```r
k_any(x, axis = -1)
```

```
## tf.Tensor([False  True], shape=(2), dtype=bool)
```

`keepdims = TRUE` outputs a tensor with dimensions reduced to one.

```r
k_any(x, keepdims = TRUE)
```

```
## tf.Tensor([[ True]], shape=(1, 1), dtype=bool)
```

```r
k_any(x, 1, keepdims = TRUE)
```

```
## tf.Tensor([[ True False False]], shape=(1, 3), dtype=bool)
```

@returns
The tensor containing the logical OR reduction over the `axis`.

@param x Input tensor.
@param axis An integer or tuple of integers that represent the axis along
    which a logical OR reduction is performed. The default
    (`axis = NULL`) is to perform a logical OR over all the dimensions
    of the input array. `axis` may be negative, in which case it counts
    for the last to the first axis.
@param keepdims If `TRUE`, axes which are reduced are left in the result as
    dimensions with size one. With this option, the result will
    broadcast correctly against the input array. Defaults to `FALSE`.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/numpy#any-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/any>
