# Initializer that adapts its scale to the shape of its input tensors.

With `distribution = "truncated_normal" or "untruncated_normal"`,
samples are drawn from a truncated/untruncated normal distribution with
a mean of zero and a standard deviation (after truncation, if used)
`stddev = sqrt(scale / n)`, where `n` is:

- number of input units in the weight tensor, if `mode = "fan_in"`

- number of output units, if `mode = "fan_out"`

- average of the numbers of input and output units, if
  `mode = "fan_avg"`

With `distribution = "uniform"`, samples are drawn from a uniform
distribution within `[-limit, limit]`, where
`limit = sqrt(3 * scale / n)`.

## Usage

``` r
initializer_variance_scaling(
  scale = 1,
  mode = "fan_in",
  distribution = "truncated_normal",
  seed = NULL
)
```

## Arguments

- scale:

  Scaling factor (positive float).

- mode:

  One of `"fan_in"`, `"fan_out"`, `"fan_avg"`.

- distribution:

  Random distribution to use. One of `"truncated_normal"`,
  `"untruncated_normal"`, or `"uniform"`.

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
    initializer <- initializer_variance_scaling(scale = 0.1, mode = 'fan_in',
                                                distribution = 'uniform')
    values <- initializer(shape = c(2, 2))

    # Usage in a Keras layer:
    initializer <- initializer_variance_scaling(scale = 0.1, mode = 'fan_in',
                                                distribution = 'uniform')
    layer <- layer_dense(units = 3, kernel_initializer = initializer)

## See also

- <https://keras.io/api/layers/initializers#variancescaling-class>

Other random initializers:  
[`initializer_glorot_normal()`](https://keras3.posit.co/dev/reference/initializer_glorot_normal.md)  
[`initializer_glorot_uniform()`](https://keras3.posit.co/dev/reference/initializer_glorot_uniform.md)  
[`initializer_he_normal()`](https://keras3.posit.co/dev/reference/initializer_he_normal.md)  
[`initializer_he_uniform()`](https://keras3.posit.co/dev/reference/initializer_he_uniform.md)  
[`initializer_lecun_normal()`](https://keras3.posit.co/dev/reference/initializer_lecun_normal.md)  
[`initializer_lecun_uniform()`](https://keras3.posit.co/dev/reference/initializer_lecun_uniform.md)  
[`initializer_orthogonal()`](https://keras3.posit.co/dev/reference/initializer_orthogonal.md)  
[`initializer_random_normal()`](https://keras3.posit.co/dev/reference/initializer_random_normal.md)  
[`initializer_random_uniform()`](https://keras3.posit.co/dev/reference/initializer_random_uniform.md)  
[`initializer_truncated_normal()`](https://keras3.posit.co/dev/reference/initializer_truncated_normal.md)  

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
[`initializer_random_normal()`](https://keras3.posit.co/dev/reference/initializer_random_normal.md)  
[`initializer_random_uniform()`](https://keras3.posit.co/dev/reference/initializer_random_uniform.md)  
[`initializer_stft()`](https://keras3.posit.co/dev/reference/initializer_stft.md)  
[`initializer_truncated_normal()`](https://keras3.posit.co/dev/reference/initializer_truncated_normal.md)  
[`initializer_zeros()`](https://keras3.posit.co/dev/reference/initializer_zeros.md)  
