keras.initializers.RandomNormal
__signature__
(mean=0.0, stddev=0.05, seed=None)
__doc__
Random normal initializer.

Draws samples from a normal distribution for given parameters.

Examples:

>>> # Standalone usage:
>>> initializer = RandomNormal(mean=0.0, stddev=1.0)
>>> values = initializer(shape=(2, 2))

>>> # Usage in a Keras layer:
>>> initializer = RandomNormal(mean=0.0, stddev=1.0)
>>> layer = Dense(3, kernel_initializer=initializer)

Args:
    mean: A python scalar or a scalar keras tensor. Mean of the random
        values to generate.
    stddev: A python scalar or a scalar keras tensor. Standard deviation of
       the random values to generate.
    seed: A Python integer or instance of
        `keras.backend.SeedGenerator`.
        Used to make the behavior of the initializer
        deterministic. Note that an initializer seeded with an integer
        or `None` (unseeded) will produce the same random values
        across multiple calls. To get different random values
        across multiple calls, use as seed an instance
        of `keras.backend.SeedGenerator`.
