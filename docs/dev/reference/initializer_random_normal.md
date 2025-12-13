# Random normal initializer.

Draws samples from a normal distribution for given parameters.

## Usage

``` r
initializer_random_normal(mean = 0, stddev = 0.05, seed = NULL)
```

## Arguments

- mean:

  A numeric scalar. Mean of the random values to generate.

- stddev:

  A numeric scalar. Standard deviation of the random values to generate.

- seed:

  An integer or instance of
  [`random_seed_generator()`](https://keras3.posit.co/dev/reference/random_seed_generator.md).
  Used to make the behavior of the initializer deterministic. Note that
  an initializer seeded with an integer or `NULL` (unseeded) will
  produce the same random values across multiple calls. To get different
  random values across multiple calls, use as seed an instance of
  [`random_seed_generator()`](https://keras3.posit.co/dev/reference/random_seed_generator.md).

## Value

An `Initializer` instance that can be passed to layer or variable
constructors, or called directly with a `shape` to return a Tensor.

## Examples

    # Standalone usage:
    initializer <- initializer_random_normal(mean = 0.0, stddev = 1.0)
    values <- initializer(shape = c(2, 2))

    # Usage in a Keras layer:
    initializer <- initializer_random_normal(mean = 0.0, stddev = 1.0)
    layer <- layer_dense(units = 3, kernel_initializer = initializer)

## See also

- <https://keras.io/api/layers/initializers#randomnormal-class>

Other random initializers:  
[`initializer_glorot_normal()`](https://keras3.posit.co/dev/reference/initializer_glorot_normal.md)  
[`initializer_glorot_uniform()`](https://keras3.posit.co/dev/reference/initializer_glorot_uniform.md)  
[`initializer_he_normal()`](https://keras3.posit.co/dev/reference/initializer_he_normal.md)  
[`initializer_he_uniform()`](https://keras3.posit.co/dev/reference/initializer_he_uniform.md)  
[`initializer_lecun_normal()`](https://keras3.posit.co/dev/reference/initializer_lecun_normal.md)  
[`initializer_lecun_uniform()`](https://keras3.posit.co/dev/reference/initializer_lecun_uniform.md)  
[`initializer_orthogonal()`](https://keras3.posit.co/dev/reference/initializer_orthogonal.md)  
[`initializer_random_uniform()`](https://keras3.posit.co/dev/reference/initializer_random_uniform.md)  
[`initializer_truncated_normal()`](https://keras3.posit.co/dev/reference/initializer_truncated_normal.md)  
[`initializer_variance_scaling()`](https://keras3.posit.co/dev/reference/initializer_variance_scaling.md)  

Other initializers:  
[`initializer_constant()`](https://keras3.posit.co/dev/reference/initializer_constant.md)  
[`initializer_glorot_normal()`](https://keras3.posit.co/dev/reference/initializer_glorot_normal.md)  
[`initializer_glorot_uniform()`](https://keras3.posit.co/dev/reference/initializer_glorot_uniform.md)  
[`initializer_he_normal()`](https://keras3.posit.co/dev/reference/initializer_he_normal.md)  
[`initializer_he_uniform()`](https://keras3.posit.co/dev/reference/initializer_he_uniform.md)  
[`initializer_identity()`](https://keras3.posit.co/dev/reference/initializer_identity.md)  
[`initializer_lecun_normal()`](https://keras3.posit.co/dev/reference/initializer_lecun_normal.md)  
[`initializer_lecun_uniform()`](https://keras3.posit.co/dev/reference/initializer_lecun_uniform.md)  
[`initializer_ones()`](https://keras3.posit.co/dev/reference/initializer_ones.md)  
[`initializer_orthogonal()`](https://keras3.posit.co/dev/reference/initializer_orthogonal.md)  
[`initializer_random_uniform()`](https://keras3.posit.co/dev/reference/initializer_random_uniform.md)  
[`initializer_stft()`](https://keras3.posit.co/dev/reference/initializer_stft.md)  
[`initializer_truncated_normal()`](https://keras3.posit.co/dev/reference/initializer_truncated_normal.md)  
[`initializer_variance_scaling()`](https://keras3.posit.co/dev/reference/initializer_variance_scaling.md)  
[`initializer_zeros()`](https://keras3.posit.co/dev/reference/initializer_zeros.md)  
