# Draws samples from a categorical distribution.

This function takes as input `logits`, a 2-D input tensor with shape
(batch_size, num_classes). Each row of the input represents a
categorical distribution, with each column index containing the
log-probability for a given class.

The function will output a 2-D tensor with shape (batch_size,
num_samples), where each row contains samples from the corresponding row
in `logits`. Each column index contains an independent samples drawn
from the input distribution.

    x <- matrix(c(100, .1, 99), nrow = 1)
    random_categorical(x, num_samples = 5, seed = 1234)

    ## tf.Tensor([[3 1 1 3 3]], shape=(1, 5), dtype=int32)

    random_categorical(x, num_samples = 5, seed = 1234,
                       zero_indexed = TRUE)

    ## tf.Tensor([[2 0 0 2 2]], shape=(1, 5), dtype=int32)

    op_take(x, random_categorical(x, num_samples = 5, seed = 1234))

    ## tf.Tensor([[ 99. 100. 100.  99.  99.]], shape=(1, 5), dtype=float64)

    op_take(x, random_categorical(x, num_samples = 5, seed = 1234,
                                  zero_indexed = TRUE),
            zero_indexed = TRUE)

    ## tf.Tensor([[ 99. 100. 100.  99.  99.]], shape=(1, 5), dtype=float64)

## Usage

``` r
random_categorical(
  logits,
  num_samples,
  dtype = "int32",
  seed = NULL,
  zero_indexed = FALSE
)
```

## Arguments

- logits:

  2-D Tensor with shape (batch_size, num_classes). Each row should
  define a categorical distribution with the unnormalized
  log-probabilities for all classes.

- num_samples:

  Int, the number of independent samples to draw for each row of the
  input. This will be the second dimension of the output tensor's shape.

- dtype:

  Optional dtype of the output tensor.

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

- zero_indexed:

  If `TRUE`, the returned indices are zero-based (`0` encodes to first
  position); if `FALSE` (default), the returned indices are one-based
  (`1` encodes to first position).

## Value

A 2-D tensor with (batch_size, num_samples).

## See also

Other random:  
[`random_beta()`](https://keras3.posit.co/reference/random_beta.md)  
[`random_binomial()`](https://keras3.posit.co/reference/random_binomial.md)  
[`random_dropout()`](https://keras3.posit.co/reference/random_dropout.md)  
[`random_gamma()`](https://keras3.posit.co/reference/random_gamma.md)  
[`random_integer()`](https://keras3.posit.co/reference/random_integer.md)  
[`random_normal()`](https://keras3.posit.co/reference/random_normal.md)  
[`random_seed_generator()`](https://keras3.posit.co/reference/random_seed_generator.md)  
[`random_shuffle()`](https://keras3.posit.co/reference/random_shuffle.md)  
[`random_truncated_normal()`](https://keras3.posit.co/reference/random_truncated_normal.md)  
[`random_uniform()`](https://keras3.posit.co/reference/random_uniform.md)  
