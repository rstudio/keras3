The Glorot normal initializer, also called Xavier normal initializer.

@description
Draws samples from a truncated normal distribution centered on 0 with
`stddev = sqrt(2 / (fan_in + fan_out))` where `fan_in` is the number of
input units in the weight tensor and `fan_out` is the number of output units
in the weight tensor.

# Examples

```r
# Standalone usage:
initializer <- initializer_glorot_normal()
values <- initializer(shape = c(2, 2))
```


```r
# Usage in a Keras layer:
initializer <- initializer_glorot_normal()
layer <- layer_dense(units = 3, kernel_initializer = initializer)
```

# Reference
- [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)

@param seed An integer or instance of
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
+ <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotNormal>
