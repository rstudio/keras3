Broadcast a tensor to a new shape.

# Returns
A tensor with the desired shape.

# Examples
```python
x = keras.ops.array([1, 2, 3])
keras.ops.broadcast_to(x, (3, 3))
# array([[1, 2, 3],
#        [1, 2, 3],
#        [1, 2, 3]])
```

@param x The tensor to broadcast.
@param shape The shape of the desired tensor. A single integer `i` is
    interpreted as `(i,)`.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/broadcast_to>
