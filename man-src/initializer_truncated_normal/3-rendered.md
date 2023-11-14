Initializer that generates a truncated normal distribution.

@description
The values generated are similar to values from a
`RandomNormal` initializer, except that values more
than two standard deviations from the mean are
discarded and re-drawn.

# Examples

```r
# Standalone usage:
initializer <- initializer_truncated_normal(mean = 0, stddev = 1)
values <- initializer(shape = c(2, 2))
```


```r
# Usage in a Keras layer:
initializer <- initializer_truncated_normal(mean = 0, stddev = 1)
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
+ <https:/keras.io/api/layers/initializers#truncatednormal-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/TruncatedNormal>
