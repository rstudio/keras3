keras.initializers.RandomUniform
__signature__
(minval=-0.05, maxval=0.05, seed=None)
__doc__
Random uniform initializer.

Draws samples from a uniform distribution for given parameters.

Examples:

>>> # Standalone usage:
>>> initializer = RandomUniform(minval=0.0, maxval=1.0)
>>> values = initializer(shape=(2, 2))

>>> # Usage in a Keras layer:
>>> initializer = RandomUniform(minval=0.0, maxval=1.0)
>>> layer = Dense(3, kernel_initializer=initializer)

Args:
    minval: A python scalar or a scalar keras tensor. Lower bound of the
        range of random values to generate (inclusive).
    maxval: A python scalar or a scalar keras tensor. Upper bound of the
        range of random values to generate (exclusive).
    seed: A Python integer or instance of
        `keras.backend.SeedGenerator`.
        Used to make the behavior of the initializer
        deterministic. Note that an initializer seeded with an integer
        or `None` (unseeded) will produce the same random values
        across multiple calls. To get different random values
        across multiple calls, use as seed an instance
        of `keras.backend.SeedGenerator`.
