Random normal initializer.

@description
Draws samples from a normal distribution for given parameters.

# Examples

```r
# Standalone usage:
initializer <- initializer_random_normal(mean = 0.0, stddev = 1.0)
values <- initializer(shape = c(2, 2))
```


```r
# Usage in a Keras layer:
initializer <- initializer_random_normal(mean = 0.0, stddev = 1.0)
layer <- layer_dense(units = 3, kernel_initializer = initializer)
```

@param mean
A numeric scalar. Mean of the random
values to generate.

@param stddev
A numeric scalar. Standard deviation of
the random values to generate.

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
+ <https:/keras.io/api/layers/initializers#randomnormal-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/RandomNormal>
