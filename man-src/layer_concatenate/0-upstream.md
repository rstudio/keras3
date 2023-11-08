keras.layers.Concatenate
__signature__
(axis=-1, **kwargs)
__doc__
Concatenates a list of inputs.

It takes as input a list of tensors, all of the same shape except
for the concatenation axis, and returns a single tensor that is the
concatenation of all inputs.

Examples:

>>> x = np.arange(20).reshape(2, 2, 5)
>>> y = np.arange(20, 30).reshape(2, 1, 5)
>>> keras.layers.Concatenate(axis=1)([x, y])

Usage in a Keras model:

>>> x1 = keras.layers.Dense(8)(np.arange(10).reshape(5, 2))
>>> x2 = keras.layers.Dense(8)(np.arange(10, 20).reshape(5, 2))
>>> y = keras.layers.Concatenate()([x1, x2])

Args:
    axis: Axis along which to concatenate.
    **kwargs: Standard layer keyword arguments.

Returns:
    A tensor, the concatenation of the inputs alongside axis `axis`.
