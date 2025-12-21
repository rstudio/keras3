# Shuffle the elements of a tensor uniformly at random along an axis.

Shuffle the elements of a tensor uniformly at random along an axis.

## Usage

``` r
random_shuffle(x, axis = 1L, seed = NULL)
```

## Arguments

- x:

  The tensor to be shuffled.

- axis:

  An integer specifying the axis along which to shuffle. Defaults to
  `0`.

- seed:

  Optional R integer or instance of
  [`random_seed_generator()`](https://keras3.posit.co/reference/random_seed_generator.md).
  By default, the `seed` argument is `NULL`, and an internal global
  [`random_seed_generator()`](https://keras3.posit.co/reference/random_seed_generator.md)
  is used. The `seed` argument can be used to ensure deterministic
  (repeatable) random number generation. Note that passing an integer as
  the `seed` value will produce the same random values for each call. To
  generate different random values for repeated calls, an instance of
  [`random_seed_generator()`](https://keras3.posit.co/reference/random_seed_generator.md)
  must be provided as the `seed` value.

  Remark concerning the JAX backend: When tracing functions with the JAX
  backend the global
  [`random_seed_generator()`](https://keras3.posit.co/reference/random_seed_generator.md)
  is not supported. Therefore, during tracing the default value
  `seed=NULL` will produce an error, and a `seed` argument must be
  provided.

## Value

A tensor, a copy of `x` with the `axis` axis shuffled.

## See also

Other random:  
[`random_beta()`](https://keras3.posit.co/reference/random_beta.md)  
[`random_binomial()`](https://keras3.posit.co/reference/random_binomial.md)  
[`random_categorical()`](https://keras3.posit.co/reference/random_categorical.md)  
[`random_dropout()`](https://keras3.posit.co/reference/random_dropout.md)  
[`random_gamma()`](https://keras3.posit.co/reference/random_gamma.md)  
[`random_integer()`](https://keras3.posit.co/reference/random_integer.md)  
[`random_normal()`](https://keras3.posit.co/reference/random_normal.md)  
[`random_seed_generator()`](https://keras3.posit.co/reference/random_seed_generator.md)  
[`random_truncated_normal()`](https://keras3.posit.co/reference/random_truncated_normal.md)  
[`random_uniform()`](https://keras3.posit.co/reference/random_uniform.md)  
