keras.initializers.GlorotUniform
__signature__
(seed=None)
__doc__
The Glorot uniform initializer, also called Xavier uniform initializer.

Draws samples from a uniform distribution within `[-limit, limit]`, where
`limit = sqrt(6 / (fan_in + fan_out))` (`fan_in` is the number of input
units in the weight tensor and `fan_out` is the number of output units).

Examples:

>>> # Standalone usage:
>>> initializer = GlorotUniform()
>>> values = initializer(shape=(2, 2))

>>> # Usage in a Keras layer:
>>> initializer = GlorotUniform()
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

- [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)
