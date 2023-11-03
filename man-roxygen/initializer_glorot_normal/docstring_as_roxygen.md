The Glorot normal initializer, also called Xavier normal initializer.

@description
Draws samples from a truncated normal distribution centered on 0 with
`stddev = sqrt(2 / (fan_in + fan_out))` where `fan_in` is the number of
input units in the weight tensor and `fan_out` is the number of output units
in the weight tensor.

# Examples
```python
# Standalone usage:
initializer = GlorotNormal()
values = initializer(shape=(2, 2))
```

```python
# Usage in a Keras layer:
initializer = GlorotNormal()
layer = Dense(3, kernel_initializer=initializer)
```

# Reference
- [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)

@param seed A Python integer or instance of
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
+ <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotNormal>
