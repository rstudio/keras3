keras.ops.shape
__signature__
(x)
__doc__
Gets the shape of the tensor input.

Note: On the TensorFlow backend, when `x` is a `tf.Tensor` with dynamic
shape, dimensions which are dynamic in the context of a compiled function
will have a `tf.Tensor` value instead of a static integer value.

Args:
    x: A tensor. This function will try to access the `shape` attribute of
        the input tensor.

Returns:
    A tuple of integers or None values, indicating the shape of the input
        tensor.

Example:

>>> x = keras.zeros((8, 12))
>>> keras.ops.shape(x)
(8, 12)
