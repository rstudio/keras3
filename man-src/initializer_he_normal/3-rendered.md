He normal initializer.

@description
It draws samples from a truncated normal distribution centered on 0 with
`stddev = sqrt(2 / fan_in)` where `fan_in` is the number of input units in
the weight tensor.

# Examples

```r
# Standalone usage:
initializer <- initializer_he_normal()
values <- initializer(shape = c(2, 2))
```


```r
# Usage in a Keras layer:
initializer <- initializer_he_normal()
layer <- layer_dense(units = 3, kernel_initializer = initializer)
```

# Reference
- [He et al., 2015](https://arxiv.org/abs/1502.01852)

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
+ <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeNormal>
