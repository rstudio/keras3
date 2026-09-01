# Logits distillation loss

Transfers knowledge from final model outputs. This loss applies
temperature scaling to the teacher logits before computing the loss
between teacher and student predictions. It is the most common approach
to knowledge distillation.

## Usage

``` r
distillation_logits(temperature = 3, loss = "kl_divergence")
```

## Arguments

- temperature:

  Temperature used for softmax scaling. Higher values produce softer
  probability distributions that are easier for the student to learn.
  Typical values range from 3 to 5. Defaults to 3.

- loss:

  Loss function used for distillation. This can be a string identifier
  such as `"kl_divergence"` or `"categorical_crossentropy"`; a Keras
  loss instance; or a nested list of losses matching the model-output
  structure. Use `NULL` within a list to skip distillation for that
  output. At least one loss must be non-`NULL`. Defaults to
  `"kl_divergence"`.

## Value

A `LogitsDistillation` instance.

## Examples

    loss <- distillation_logits(temperature = 3)

## See also

Other distillation:  
[`distillation_feature()`](https://keras3.posit.co/dev/reference/distillation_feature.md)  
[`distiller()`](https://keras3.posit.co/dev/reference/distiller.md)  
