# Base class for distillation loss computation

Distillation losses define how to compute the loss between teacher and
student outputs. Each loss implements a specific approach to knowledge
transfer, from logits matching to feature-based distillation. Custom
distillation losses can subclass this class and implement
`compute_loss()`. This base class does not implement a loss; use
[`distillation_feature()`](https://keras3.posit.co/dev/reference/distillation_feature.md)
or
[`distillation_logits()`](https://keras3.posit.co/dev/reference/distillation_logits.md)
directly.

## Usage

``` r
distillation_loss()
```

## Value

A `DistillationLoss` instance.

## See also

Other distillation:  
[`distillation_feature()`](https://keras3.posit.co/dev/reference/distillation_feature.md)  
[`distillation_logits()`](https://keras3.posit.co/dev/reference/distillation_logits.md)  
[`distiller()`](https://keras3.posit.co/dev/reference/distiller.md)  
