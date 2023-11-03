Broadcast a tensor to a new shape.

Args:
    x: The tensor to broadcast.
    shape: The shape of the desired tensor. A single integer `i` is
        interpreted as `(i,)`.

Returns:
    A tensor with the desired shape.

Examples:
>>> x = keras.ops.array([1, 2, 3])
>>> keras.ops.broadcast_to(x, (3, 3))
array([[1, 2, 3],
       [1, 2, 3],
       [1, 2, 3]])
