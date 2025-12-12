# Draw samples from a truncated normal distribution.

The values are drawn from a normal distribution with specified mean and
standard deviation, discarding and re-drawing any samples that are more
than two standard deviations from the mean.

## Usage

``` r
random_truncated_normal(shape, mean = 0, stddev = 1, dtype = NULL, seed = NULL)
```

## Arguments

- shape:

  The shape of the random values to generate.

- mean:

  Float, defaults to 0. Mean of the random values to generate.

- stddev:

  Float, defaults to 1. Standard deviation of the random values to
  generate.

- dtype:

  Optional dtype of the tensor. Only floating point types are supported.
  If not specified,
  [`config_floatx()`](https://keras3.posit.co/dev/reference/config_floatx.md)
  is used, which defaults to `float32` unless you configured it
  otherwise (via `config_set_floatx(float_dtype)`)

- seed:

  Optional R integer or instance of
  [`random_seed_generator()`](https://keras3.posit.co/dev/reference/random_seed_generator.md).
  By default, the `seed` argument is `NULL`, and an internal global
  [`random_seed_generator()`](https://keras3.posit.co/dev/reference/random_seed_generator.md)
  is used. The `seed` argument can be used to ensure deterministic
  (repeatable) random number generation. Note that passing an integer as
  the `seed` value will produce the same random values for each call. To
  generate different random values for repeated calls, an instance of
  [`random_seed_generator()`](https://keras3.posit.co/dev/reference/random_seed_generator.md)
  must be provided as the `seed` value.

  Remark concerning the JAX backend: When tracing functions with the JAX
  backend the global
  [`random_seed_generator()`](https://keras3.posit.co/dev/reference/random_seed_generator.md)
  is not supported. Therefore, during tracing the default value
  `seed=NULL` will produce an error, and a `seed` argument must be
  provided.

## Value

A tensor of random values.

## See also

Other random:  
[`random_beta()`](https://keras3.posit.co/dev/reference/random_beta.md)  
[`random_binomial()`](https://keras3.posit.co/dev/reference/random_binomial.md)  
[`random_categorical()`](https://keras3.posit.co/dev/reference/random_categorical.md)  
[`random_dropout()`](https://keras3.posit.co/dev/reference/random_dropout.md)  
[`random_gamma()`](https://keras3.posit.co/dev/reference/random_gamma.md)  
[`random_integer()`](https://keras3.posit.co/dev/reference/random_integer.md)  
[`random_normal()`](https://keras3.posit.co/dev/reference/random_normal.md)  
[`random_seed_generator()`](https://keras3.posit.co/dev/reference/random_seed_generator.md)  
[`random_shuffle()`](https://keras3.posit.co/dev/reference/random_shuffle.md)  
[`random_uniform()`](https://keras3.posit.co/dev/reference/random_uniform.md)  
