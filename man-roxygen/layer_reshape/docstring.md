Layer that reshapes inputs into the given shape.

Args:
    target_shape: Target shape. Tuple of integers, does not include the
        samples dimension (batch size).

Input shape:
    Arbitrary, although all dimensions in the input shape must be
    known/fixed. Use the keyword argument `input_shape` (tuple of integers,
    does not include the samples/batch size axis) when using this layer as
    the first layer in a model.

Output shape:
    `(batch_size, *target_shape)`

Example:

>>> x = keras.Input(shape=(12,))
>>> y = keras.layers.Reshape((3, 4))(x)
>>> y.shape
(None, 3, 4)

>>> # also supports shape inference using `-1` as dimension
>>> y = keras.layers.Reshape((-1, 2, 2))(x)
>>> y.shape
(None, 3, 2, 2)
