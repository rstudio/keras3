Initializer that generates tensors initialized to 1.

Also available via the shortcut function `ones`.

Examples:

>>> # Standalone usage:
>>> initializer = Ones()
>>> values = initializer(shape=(2, 2))

>>> # Usage in a Keras layer:
>>> initializer = Ones()
>>> layer = Dense(3, kernel_initializer=initializer)
