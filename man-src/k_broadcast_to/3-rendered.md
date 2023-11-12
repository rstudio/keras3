Broadcast a tensor to a new shape.

@description

# Examples

```r
x <- k_array(c(1, 2, 3))
k_broadcast_to(x, shape = c(3, 3))
```

```
## tf.Tensor(
## [[1. 2. 3.]
##  [1. 2. 3.]
##  [1. 2. 3.]], shape=(3, 3), dtype=float32)
```

@returns
A tensor with the desired shape.

@param x The tensor to broadcast.
@param shape The shape of the desired tensor.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/numpy#broadcastto-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/broadcast_to>
