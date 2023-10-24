Lecun uniform initializer.

Draws samples from a uniform distribution within `[-limit, limit]`, where
`limit = sqrt(3 / fan_in)` (`fan_in` is the number of input units in the
weight tensor).

Examples:

>>> # Standalone usage:
>>> initializer = LecunUniform()
>>> values = initializer(shape=(2, 2))

>>> # Usage in a Keras layer:
>>> initializer = LecunUniform()
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

- [Klambauer et al., 2017](https://arxiv.org/abs/1706.02515)
