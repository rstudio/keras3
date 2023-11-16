keras.random.normal
__signature__
(
  shape,
  mean=0.0,
  stddev=1.0,
  dtype=None,
  seed=None
)
__doc__
Draw random samples from a normal (Gaussian) distribution.

Args:
    shape: The shape of the random values to generate.
    mean: Float, defaults to 0. Mean of the random values to generate.
    stddev: Float, defaults to 1. Standard deviation of the random values
        to generate.
    dtype: Optional dtype of the tensor. Only floating point types are
        supported. If not specified, `keras.config.floatx()` is used,
        which defaults to `float32` unless you configured it otherwise (via
        `keras.config.set_floatx(float_dtype)`).
    seed: A Python integer or instance of
        `keras.random.SeedGenerator`.
        Used to make the behavior of the initializer
        deterministic. Note that an initializer seeded with an integer
        or None (unseeded) will produce the same random values
        across multiple calls. To get different random values
        across multiple calls, use as seed an instance
        of `keras.random.SeedGenerator`.
