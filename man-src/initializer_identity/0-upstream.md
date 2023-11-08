keras.initializers.IdentityInitializer
__signature__
(gain=1.0)
__doc__
Initializer that generates the identity matrix.

Only usable for generating 2D matrices.

Examples:

>>> # Standalone usage:
>>> initializer = Identity()
>>> values = initializer(shape=(2, 2))

>>> # Usage in a Keras layer:
>>> initializer = Identity()
>>> layer = Dense(3, kernel_initializer=initializer)

Args:
    gain: Multiplicative factor to apply to the identity matrix.
