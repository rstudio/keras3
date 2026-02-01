# Randomly set some values in a tensor to 0.

Randomly set some portion of values in the tensor to 0.

## Usage

``` r
random_dropout(inputs, rate, noise_shape = NULL, seed = NULL)
```

## Arguments

- inputs:

  A tensor

- rate:

  numeric

- noise_shape:

  A [`shape()`](https://keras3.posit.co/reference/shape.md) value

- seed:

  Initial seed for the random number generator

## Value

A tensor that is a copy of `inputs` with some values set to `0`.

## See also

Other random:  
[`random_beta()`](https://keras3.posit.co/reference/random_beta.md)  
[`random_binomial()`](https://keras3.posit.co/reference/random_binomial.md)  
[`random_categorical()`](https://keras3.posit.co/reference/random_categorical.md)  
[`random_gamma()`](https://keras3.posit.co/reference/random_gamma.md)  
[`random_integer()`](https://keras3.posit.co/reference/random_integer.md)  
[`random_normal()`](https://keras3.posit.co/reference/random_normal.md)  
[`random_seed_generator()`](https://keras3.posit.co/reference/random_seed_generator.md)  
[`random_shuffle()`](https://keras3.posit.co/reference/random_shuffle.md)  
[`random_truncated_normal()`](https://keras3.posit.co/reference/random_truncated_normal.md)  
[`random_uniform()`](https://keras3.posit.co/reference/random_uniform.md)  
