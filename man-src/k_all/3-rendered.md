Test whether all array elements along a given axis evaluate to `TRUE`.

@description

# Examples

```r
x <- k_convert_to_tensor(c(TRUE, FALSE))
k_all(x)
```

```
## tf.Tensor(False, shape=(), dtype=bool)
```


```r
(x <- k_convert_to_tensor(array(c(TRUE, FALSE, TRUE, TRUE, TRUE, TRUE), dim = c(3, 2))))
```

```
## tf.Tensor(
## [[ True  True]
##  [False  True]
##  [ True  True]], shape=(3, 2), dtype=bool)
```

```r
k_all(x, axis = 1)
```

```
## tf.Tensor([False  True], shape=(2), dtype=bool)
```

`keepdims = TRUE` outputs a tensor with dimensions reduced to one.

```r
k_all(x, keepdims = TRUE)
```

```
## tf.Tensor([[False]], shape=(1, 1), dtype=bool)
```

@returns
The tensor containing the logical AND reduction over the `axis`.

@param x
Input tensor.

@param axis
An integer or tuple of integers that represent the axis along
which a logical AND reduction is performed. The default
(`axis = NULL`) is to perform a logical AND over all the dimensions
of the input array. `axis` may be negative, in which case it counts
for the last to the first axis.

@param keepdims
If `TRUE`, axes which are reduced are left in the result as
dimensions with size one. With this option, the result will
broadcast correctly against the input array. Defaults to`FALSE`.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/numpy#all-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/all>
