keras.initializers.HeUniform
__signature__
(seed=None)
__doc__
He uniform variance scaling initializer.

Draws samples from a uniform distribution within `[-limit, limit]`, where
`limit = sqrt(6 / fan_in)` (`fan_in` is the number of input units in the
weight tensor).

Examples:

>>> # Standalone usage:
>>> initializer = HeUniform()
>>> values = initializer(shape=(2, 2))

>>> # Usage in a Keras layer:
>>> initializer = HeUniform()
>>> layer = Dense(3, kernel_initializer=initializer)

Args:
    seed: A Python integer or instance of
        `keras.backend.SeedGenerator`.
        Used to make the behavior of the initializer
        deterministic. Note that an initializer seeded with an integer
        or `None` (unseeded) will produce the same random values
        across multiple calls. To get different random values
        across multiple calls, use as seed an instance
        of `keras.backend.SeedGenerator`.

Reference:

- [He et al., 2015](https://arxiv.org/abs/1502.01852)
