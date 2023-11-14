Random uniform initializer.

@description
Draws samples from a uniform distribution for given parameters.

# Examples

```r
# Standalone usage:
initializer <- initializer_random_uniform(minval = 0.0, maxval = 1.0)
values <- initializer(shape = c(2, 2))
```


```r
# Usage in a Keras layer:
initializer <- initializer_random_uniform(minval = 0.0, maxval = 1.0)
layer <- layer_dense(units = 3, kernel_initializer = initializer)
```

@param minval
A numeric scalar or a scalar keras tensor. Lower bound of the
range of random values to generate (inclusive).

@param maxval
A numeric scalar or a scalar keras tensor. Upper bound of the
range of random values to generate (exclusive).

@param seed
An integer or instance of
`random_seed_generator()`.
Used to make the behavior of the initializer
deterministic. Note that an initializer seeded with an integer
or `NULL` (unseeded) will produce the same random values
across multiple calls. To get different random values
across multiple calls, use as seed an instance
of `random_seed_generator()`.

@export
@family initializer
@seealso
+ <https:/keras.io/api/layers/initializers#randomuniform-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/RandomUniform>
