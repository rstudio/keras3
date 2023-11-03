Cast a tensor to the desired dtype.

Args:
    x: A tensor or variable.
    dtype: The target type.

Returns:
    A tensor of the specified `dtype`.

Example:

>>> x = keras.ops.arange(4)
>>> x = keras.ops.cast(x, dtype="float16")
