Add arguments element-wise.

@description

# Examples

```r
x1 <- k_convert_to_tensor(c(1, 4))
x2 <- k_convert_to_tensor(c(5, 6))
k_add(x1, x2)
```

```
## tf.Tensor([ 6. 10.], shape=(2), dtype=float32)
```

```r
# alias for x1 + x2
x1 + x2
```

```
## tf.Tensor([ 6. 10.], shape=(2), dtype=float32)
```

`k_add` also broadcasts shapes:

```r
x1 <- k_convert_to_tensor(array(c(5, 5, 4, 6), dim =c(2, 2)))
x2 <- k_convert_to_tensor(c(5, 6))
k_add(x1, x2)
```

```
## tf.Tensor(
## [[10. 10.]
##  [10. 12.]], shape=(2, 2), dtype=float64)
```

@returns
The tensor containing the element-wise sum of `x1` and `x2`.

@param x1 First input tensor.
@param x2 Second input tensor.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/numpy#add-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/add>
