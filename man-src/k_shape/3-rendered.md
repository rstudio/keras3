Gets the shape of the tensor input.

@description

# Note
On the tensorflow backend, when `x` is a `tf.Tensor` with dynamic
shape, dimensions which are dynamic in the context of a compiled function
will have a `tf.Tensor` value instead of a static integer value.

# Examples
```python
x = keras.zeros((8, 12))
keras.ops.shape(x)
# (8, 12)
```

@returns
A tuple of integers or None values, indicating the shape of the input
tensor.

@param x A tensor. This function will try to access the `shape` attribute of
the input tensor.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/shape>
