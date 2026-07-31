# Logits distillation loss

Applies temperature scaling to teacher logits before computing the loss
between teacher and student predictions.

## Usage

``` r
distillation_logits(temperature = 3, loss = "kl_divergence")
```

## Arguments

- temperature:

  Temperature used for softmax scaling. Higher values produce softer
  probability distributions.

- loss:

  Loss function used for distillation. This can be a string identifier,
  a Keras loss instance, or a nested list matching the model output
  structure. Use `NULL` within a list to skip an output.

## Value

A `LogitsDistillation` instance.

## Examples

    loss <- distillation_logits(temperature = 3)

## See also

Other distillation:  
[`distillation_feature()`](https://keras3.posit.co/dev/reference/distillation_feature.md)  
[`distillation_loss()`](https://keras3.posit.co/dev/reference/distillation_loss.md)  
[`distiller()`](https://keras3.posit.co/dev/reference/distiller.md)  
