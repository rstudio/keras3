Random normal initializer.

@description
Draws samples from a normal distribution for given parameters.

# Examples
```python
# Standalone usage:
initializer = RandomNormal(mean=0.0, stddev=1.0)
values = initializer(shape=(2, 2))
```

```python
# Usage in a Keras layer:
initializer = RandomNormal(mean=0.0, stddev=1.0)
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
@family random initializers
@family initializers
@seealso
+ <https:/keras.io/api/layers/initializers#randomnormal-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/RandomNormal>
