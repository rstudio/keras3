keras.initializers.Zeros
__signature__
()
__doc__
Initializer that generates tensors initialized to 0.

Examples:

>>> # Standalone usage:
>>> initializer = Zeros()
>>> values = initializer(shape=(2, 2))

>>> # Usage in a Keras layer:
>>> initializer = Zeros()
>>> layer = Dense(units=3, kernel_initializer=initializer)
