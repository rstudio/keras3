# Generates variable seeds upon each call to a function generating random numbers.

In Keras, all random number generators (such as
[`random_normal()`](https://keras3.posit.co/dev/reference/random_normal.md))
are stateless, meaning that if you pass an integer seed to them (such as
`seed=42`), they will return the same values for repeated calls. To get
different values for each call, a `SeedGenerator` providing the state of
the random generator has to be used.

Note that all the random number generators have a default seed of
`NULL`, which implies that an internal global `SeedGenerator` is used.
If you need to decouple the RNG from the global state you can provide a
local `StateGenerator` with either a deterministic or random initial
state.

Remark concerning the JAX backen: Note that the use of a local
`StateGenerator` as seed argument is required for JIT compilation of RNG
with the JAX backend, because the use of global state is not supported.

## Usage

``` r
random_seed_generator(seed = NULL, name = NULL, ...)
```

## Arguments

- seed:

  Initial seed for the random number generator

- name:

  String, name for the object

- ...:

  For forward/backward compatability.

## Value

A `SeedGenerator` instance, which can be passed as the `seed = `
argument to other random tensor generators.

## Examples

    seed_gen <- random_seed_generator(seed = 42)
    values <- random_normal(shape = c(2, 3), seed = seed_gen)
    new_values <- random_normal(shape = c(2, 3), seed = seed_gen)

Usage in a layer:

    layer_dropout2 <- new_layer_class(
      "dropout2",
      initialize = function(...) {
        super$initialize(...)
        self$seed_generator <- random_seed_generator(seed = 1337)
      },
      call = function(x, training = FALSE) {
        if (training) {
          return(random_dropout(x, rate = 0.5, seed = self$seed_generator))
        }
        return(x)
      }
    )

    out <- layer_dropout(rate = 0.8)
    out(op_ones(10), training = TRUE)

    ## tf.Tensor([0. 5. 5. 0. 0. 0. 0. 0. 0. 0.], shape=(10), dtype=float32)

## See also

Other random:  
[`random_beta()`](https://keras3.posit.co/dev/reference/random_beta.md)  
[`random_binomial()`](https://keras3.posit.co/dev/reference/random_binomial.md)  
[`random_categorical()`](https://keras3.posit.co/dev/reference/random_categorical.md)  
[`random_dropout()`](https://keras3.posit.co/dev/reference/random_dropout.md)  
[`random_gamma()`](https://keras3.posit.co/dev/reference/random_gamma.md)  
[`random_integer()`](https://keras3.posit.co/dev/reference/random_integer.md)  
[`random_normal()`](https://keras3.posit.co/dev/reference/random_normal.md)  
[`random_shuffle()`](https://keras3.posit.co/dev/reference/random_shuffle.md)  
[`random_truncated_normal()`](https://keras3.posit.co/dev/reference/random_truncated_normal.md)  
[`random_uniform()`](https://keras3.posit.co/dev/reference/random_uniform.md)  
