# Map variables to optimizers

When retrieving an optimizer, the map first attempts an exact key match.
If no exact match is found, it treats every key as a regular-expression
pattern and performs a full match against the variable path. For
example, `"encoder"` matches only the layer with that exact path; use
`"encoder/.*"` to match its sublayers. Exact matches take precedence. If
more than one regular expression matches, lookup raises an error.
Variables not matched by the map use `default_optimizer`.

## Usage

``` r
optimizer_map(default_optimizer, optimizer_map = NULL)
```

## Arguments

- default_optimizer:

  Keras optimizer used for unmatched variables.

- optimizer_map:

  Optional named list mapping exact variable paths or regular
  expressions to optimizer instances.

## Value

An `OptimizerMap` instance.

## Examples

    optimizers <- optimizer_map(
      default_optimizer = optimizer_sgd(),
      optimizer_map = list("encoder/.*" = optimizer_adam())
    )

## See also

Other optimizers:  
[`optimizer_adadelta()`](https://keras3.posit.co/dev/reference/optimizer_adadelta.md)  
[`optimizer_adafactor()`](https://keras3.posit.co/dev/reference/optimizer_adafactor.md)  
[`optimizer_adagrad()`](https://keras3.posit.co/dev/reference/optimizer_adagrad.md)  
[`optimizer_adam()`](https://keras3.posit.co/dev/reference/optimizer_adam.md)  
[`optimizer_adam_w()`](https://keras3.posit.co/dev/reference/optimizer_adam_w.md)  
[`optimizer_adamax()`](https://keras3.posit.co/dev/reference/optimizer_adamax.md)  
[`optimizer_ftrl()`](https://keras3.posit.co/dev/reference/optimizer_ftrl.md)  
[`optimizer_lamb()`](https://keras3.posit.co/dev/reference/optimizer_lamb.md)  
[`optimizer_lion()`](https://keras3.posit.co/dev/reference/optimizer_lion.md)  
[`optimizer_loss_scale()`](https://keras3.posit.co/dev/reference/optimizer_loss_scale.md)  
[`optimizer_multi()`](https://keras3.posit.co/dev/reference/optimizer_multi.md)  
[`optimizer_muon()`](https://keras3.posit.co/dev/reference/optimizer_muon.md)  
[`optimizer_nadam()`](https://keras3.posit.co/dev/reference/optimizer_nadam.md)  
[`optimizer_rmsprop()`](https://keras3.posit.co/dev/reference/optimizer_rmsprop.md)  
[`optimizer_schedule_free_adam_w()`](https://keras3.posit.co/dev/reference/optimizer_schedule_free_adam_w.md)  
[`optimizer_sgd()`](https://keras3.posit.co/dev/reference/optimizer_sgd.md)  
