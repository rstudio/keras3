Draw samples from a truncated normal distribution.

The values are drawn from a normal distribution with specified mean and
standard deviation, discarding and re-drawing any samples that are more
than two standard deviations from the mean.

Args:
    shape: The shape of the random values to generate.
    mean: Floats, defaults to 0. Mean of the random values to generate.
    stddev: Floats, defaults to 1. Standard deviation of the random values
        to generate.
    dtype: Optional dtype of the tensor. Only floating point types are
        supported. If not specified, `keras.config.floatx()` is used,
        which defaults to `float32` unless you configured it otherwise (via
        `keras.config.set_floatx(float_dtype)`)
    seed: A Python integer or instance of
        `keras.random.SeedGenerator`.
        Used to make the behavior of the initializer
        deterministic. Note that an initializer seeded with an integer
        or None (unseeded) will produce the same random values
        across multiple calls. To get different random values
        across multiple calls, use as seed an instance
        of `keras.random.SeedGenerator`.
