Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.

@description

# Examples

```r
x <- k_array(rbind(c(1, 2),
                   c(3, 4)))
k_unstack(x, axis=1)
```

```
## [[1]]
## tf.Tensor([1. 2.], shape=(2), dtype=float64)
##
## [[2]]
## tf.Tensor([3. 4.], shape=(2), dtype=float64)
```

```r
k_unstack(x, axis=2)
```

```
## [[1]]
## tf.Tensor([1. 3.], shape=(2), dtype=float64)
##
## [[2]]
## tf.Tensor([2. 4.], shape=(2), dtype=float64)
```



```r
all.equal(k_unstack(x),
          k_unstack(x, axis = 1))
```

```
## [1] TRUE
```

```r
all.equal(k_unstack(x, axis = -1),
          k_unstack(x, axis = 2))
```

```
## [1] TRUE
```

```r
# [array([1, 2)), array([3, 4))]
```

@returns
A list of tensors unpacked along the given axis.

@param x
The input tensor.

@param num
The length of the dimension axis. Automatically inferred
if `NULL`.

@param axis
The axis along which to unpack.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/unstack>

