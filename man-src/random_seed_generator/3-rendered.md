Generates variable seeds upon each call to a RNG-using function.

@description
In Keras, all RNG-using methods (such as `random_normal()`)
are stateless, meaning that if you pass an integer seed to them
(such as `seed = 42`), they will return the same values at each call.
In order to get different values at each call, you must use a
`SeedGenerator` instead as the seed argument. The `SeedGenerator`
object is stateful.

# Examples

```r
seed_gen <- random_seed_generator(seed = 42)
values <- random_normal(shape = c(2, 3), seed = seed_gen)
new_values <- random_normal(shape = c(2, 3), seed = seed_gen)
```

Usage in a layer:


```r
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

out <- layer_dropout(k_ones(10), rate = 0.8)
out
```

```
## tf.Tensor([1. 1. 1. 1. 1. 1. 1. 1. 1. 1.], shape=(10), dtype=float32)
```

@param seed
Initial seed for the random number generator

@param ...
Passed on to the Python callable

@export
@family random
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/random/SeedGenerator>

