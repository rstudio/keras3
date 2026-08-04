# Delegate variables to multiple optimizers

Wraps an
[`optimizer_map()`](https://keras3.posit.co/dev/reference/optimizer_map.md)
or a callable that selects an optimizer for a variable. Access the
sub-optimizers through `optimizer$optimizers`, for example to inspect
their learning rates, iterations, or loss-scale factors. A
multi-optimizer does not expose a single `learning_rate`, because its
sub-optimizers may have different learning rates. Optimizer-specific
callbacks are not currently supported.

## Usage

``` r
optimizer_multi(optimizer_map, loss_scale_factor = NULL, name = NULL)
```

## Arguments

- optimizer_map:

  An `OptimizerMap` or callable that accepts a variable and returns its
  optimizer.

- loss_scale_factor:

  Optional loss scale overriding the value on each sub-optimizer.

- name:

  Optional optimizer name.

## Value

A `MultiOptimizer` instance.

## Examples

    optimizers <- optimizer_map(
      default_optimizer = optimizer_sgd(),
      optimizer_map = list("encoder/.*" = optimizer_adam())
    )
    optimizer <- optimizer_multi(optimizers)

    # A callable can also select an optimizer for each variable.
    optimizer <- optimizer_multi(function(variable) {
      if (grepl("encoder", variable$path)) optimizer_adam() else optimizer_sgd()
    })

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
[`optimizer_map()`](https://keras3.posit.co/dev/reference/optimizer_map.md)  
[`optimizer_muon()`](https://keras3.posit.co/dev/reference/optimizer_muon.md)  
[`optimizer_nadam()`](https://keras3.posit.co/dev/reference/optimizer_nadam.md)  
[`optimizer_rmsprop()`](https://keras3.posit.co/dev/reference/optimizer_rmsprop.md)  
[`optimizer_schedule_free_adam_w()`](https://keras3.posit.co/dev/reference/optimizer_schedule_free_adam_w.md)  
[`optimizer_sgd()`](https://keras3.posit.co/dev/reference/optimizer_sgd.md)  
