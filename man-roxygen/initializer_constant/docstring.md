Initializer that generates tensors with constant values.

Only scalar values are allowed.
The constant value provided must be convertible to the dtype requested
when calling the initializer.

Examples:

>>> # Standalone usage:
>>> initializer = Constant(10.)
>>> values = initializer(shape=(2, 2))

>>> # Usage in a Keras layer:
>>> initializer = Constant(10.)
>>> layer = Dense(3, kernel_initializer=initializer)

Args:
    value: A Python scalar.
