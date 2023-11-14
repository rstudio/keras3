Initializer that generates a truncated normal distribution.

@description
The values generated are similar to values from a
`RandomNormal` initializer, except that values more
than two standard deviations from the mean are
discarded and re-drawn.

# Examples
```python
# Standalone usage:
initializer = TruncatedNormal(mean=0., stddev=1.)
values = initializer(shape=(2, 2))
```

```python
# Usage in a Keras layer:
initializer = TruncatedNormal(mean=0., stddev=1.)
layer = Dense(3, kernel_initializer=initializer)
```

@param mean
A python scalar or a scalar keras tensor. Mean of the random
values to generate.

@param stddev
A python scalar or a scalar keras tensor. Standard deviation of
the random values to generate.

@param seed
A Python integer or instance of
`keras.backend.SeedGenerator`.
Used to make the behavior of the initializer
deterministic. Note that an initializer seeded with an integer
or `None` (unseeded) will produce the same random values
across multiple calls. To get different random values
across multiple calls, use as seed an instance
of `keras.backend.SeedGenerator`.

@export
@family initializer
@seealso
+ <https:/keras.io/api/layers/initializers#truncatednormal-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/TruncatedNormal>
