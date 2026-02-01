# Draw random integers from a uniform distribution.

The generated values follow a uniform distribution in the range
`[minval, maxval)`. The lower bound `minval` is included in the range,
while the upper bound `maxval` is excluded.

`dtype` must be an integer type.

## Usage

``` r
random_integer(shape, minval, maxval, dtype = "int32", seed = NULL)
```

## Arguments

- shape:

  The shape of the random values to generate.

- minval:

  integer, lower bound of the range of random values to generate
  (inclusive).

- maxval:

  integer, upper bound of the range of random values to generate
  (exclusive).

- dtype:

  Optional dtype of the tensor. Only integer types are supported. If not
  specified, `"int32"` is used.

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

A tensor of random values.

## See also

Other random:  
[`random_beta()`](https://keras3.posit.co/reference/random_beta.md)  
[`random_binomial()`](https://keras3.posit.co/reference/random_binomial.md)  
[`random_categorical()`](https://keras3.posit.co/reference/random_categorical.md)  
[`random_dropout()`](https://keras3.posit.co/reference/random_dropout.md)  
[`random_gamma()`](https://keras3.posit.co/reference/random_gamma.md)  
[`random_normal()`](https://keras3.posit.co/reference/random_normal.md)  
[`random_seed_generator()`](https://keras3.posit.co/reference/random_seed_generator.md)  
[`random_shuffle()`](https://keras3.posit.co/reference/random_shuffle.md)  
[`random_truncated_normal()`](https://keras3.posit.co/reference/random_truncated_normal.md)  
[`random_uniform()`](https://keras3.posit.co/reference/random_uniform.md)  
