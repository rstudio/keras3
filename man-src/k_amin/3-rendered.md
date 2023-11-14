Returns the minimum of a vector or minimum value along an axis.

@description

# Examples

```r
(x <- k_convert_to_tensor(rbind(c(1, 3, 5), c(1, 5, 2))))
```

```
## tf.Tensor(
## [[1. 3. 5.]
##  [1. 5. 2.]], shape=(2, 3), dtype=float64)
```

```r
k_amin(x)
```

```
## tf.Tensor(1.0, shape=(), dtype=float64)
```

```r
k_amin(x, axis = 1)
```

```
## tf.Tensor([1. 3. 2.], shape=(3), dtype=float64)
```

```r
k_amin(x, axis = 1, keepdims = TRUE)
```

```
## tf.Tensor([[1. 3. 2.]], shape=(1, 3), dtype=float64)
```

@returns
A tensor with the minimum value. If `axis = NULL`, the result is a scalar
value representing the minimum element in the entire tensor. If `axis` is
given, the result is a tensor with the minimum values along
the specified axis.

@param x
Input tensor.

@param axis
Axis along which to compute the minimum.
By default (`axis = NULL`), find the minimum value in all the
dimensions of the input tensor.

@param keepdims
If `TRUE`, axes which are reduced are left in the result as
dimensions that are broadcast to the size of the original
input tensor. Defaults to `FALSE`.

@export
@family numpy ops
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/numpy#amin-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/amin>
