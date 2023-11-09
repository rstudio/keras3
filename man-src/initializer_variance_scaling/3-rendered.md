Initializer that adapts its scale to the shape of its input tensors.

@description
With `distribution="truncated_normal" or "untruncated_normal"`, samples are
drawn from a truncated/untruncated normal distribution with a mean of zero
and a standard deviation (after truncation, if used) `stddev = sqrt(scale /
n)`, where `n` is:

- number of input units in the weight tensor, if `mode="fan_in"`
- number of output units, if `mode="fan_out"`
- average of the numbers of input and output units, if `mode="fan_avg"`

With `distribution="uniform"`, samples are drawn from a uniform distribution
within `[-limit, limit]`, where `limit = sqrt(3 * scale / n)`.

# Examples

```r
# Standalone usage:
initializer <- initializer_variance_scaling(scale = 0.1, mode = 'fan_in',
                                            distribution = 'uniform')
values <- initializer(shape = c(2, 2))
```


```r
# Usage in a Keras layer:
initializer <- initializer_variance_scaling(scale = 0.1, mode = 'fan_in',
                                            distribution = 'uniform')
layer <- layer_dense(units = 3, kernel_initializer = initializer)
```

@param scale Scaling factor (positive float).
@param mode One of `"fan_in"`, `"fan_out"`, `"fan_avg"`.
@param distribution Random distribution to use.
    One of `"truncated_normal"`, `"untruncated_normal"`, or `"uniform"`.
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
+ <https:/keras.io/api/layers/initializers#variancescaling-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/VarianceScaling>
