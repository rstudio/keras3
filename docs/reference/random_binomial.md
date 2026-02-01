# Draw samples from a Binomial distribution.

The values are drawn from a Binomial distribution with specified trial
count and probability of success.

## Usage

``` r
random_binomial(shape, counts, probabilities, dtype = NULL, seed = NULL)
```

## Arguments

- shape:

  The shape of the random values to generate.

- counts:

  A number or array of numbers representing the number of trials. It
  must be broadcastable with `probabilities`.

- probabilities:

  A float or array of floats representing the probability of success of
  an individual event. It must be broadcastable with `counts`.

- dtype:

  Optional dtype of the tensor. Only floating point types are supported.
  If not specified,
  [`config_floatx()`](https://keras3.posit.co/reference/config_floatx.md)
  is used, which defaults to `"float32"` unless you configured it
  otherwise (via `config_set_floatx(float_dtype)`).

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

- <https://www.tensorflow.org/api_docs/python/tf/keras/random/binomial>

Other random:  
[`random_beta()`](https://keras3.posit.co/reference/random_beta.md)  
[`random_categorical()`](https://keras3.posit.co/reference/random_categorical.md)  
[`random_dropout()`](https://keras3.posit.co/reference/random_dropout.md)  
[`random_gamma()`](https://keras3.posit.co/reference/random_gamma.md)  
[`random_integer()`](https://keras3.posit.co/reference/random_integer.md)  
[`random_normal()`](https://keras3.posit.co/reference/random_normal.md)  
[`random_seed_generator()`](https://keras3.posit.co/reference/random_seed_generator.md)  
[`random_shuffle()`](https://keras3.posit.co/reference/random_shuffle.md)  
[`random_truncated_normal()`](https://keras3.posit.co/reference/random_truncated_normal.md)  
[`random_uniform()`](https://keras3.posit.co/reference/random_uniform.md)  
