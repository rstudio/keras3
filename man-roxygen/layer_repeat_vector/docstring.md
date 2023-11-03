Repeats the input n times.

Example:

>>> x = keras.Input(shape=(32,))
>>> y = keras.layers.RepeatVector(3)(x)
>>> y.shape
(None, 3, 32)

Args:
    n: Integer, repetition factor.

Input shape:
    2D tensor with shape `(batch_size, features)`.

Output shape:
    3D tensor with shape `(batch_size, n, features)`.
