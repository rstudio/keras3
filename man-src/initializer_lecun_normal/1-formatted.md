Lecun normal initializer.

@description
Initializers allow you to pre-specify an initialization strategy, encoded in
the Initializer object, without knowing the shape and dtype of the variable
being initialized.

Draws samples from a truncated normal distribution centered on 0 with
`stddev = sqrt(1 / fan_in)` where `fan_in` is the number of input units in
the weight tensor.

# Examples
```python
# Standalone usage:
initializer = LecunNormal()
values = initializer(shape=(2, 2))
```

```python
# Usage in a Keras layer:
initializer = LecunNormal()
layer = Dense(3, kernel_initializer=initializer)
```

# Reference
- [Klambauer et al., 2017](https://arxiv.org/abs/1706.02515)

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
+ <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/LecunNormal>
