Gets the shape of the tensor input.

@description

# Note
On the tensorflow backend, when `x` is a `tf.Tensor` with dynamic
shape, dimensions which are dynamic in the context of a compiled function
will have a `tf.Tensor` value instead of a static integer value.

# Examples

```r
x <- k_zeros(c(8, 12))
k_shape(x)
```

```
## [[1]]
## [1] 8
##
## [[2]]
## [1] 12
```

@returns
A list of integers or NULL values, indicating the shape of the input
tensor.

@param x A tensor. This function will try to access the `shape` attribute of
the input tensor.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/core#shape-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/shape>
