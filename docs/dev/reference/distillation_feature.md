# Feature distillation loss

Feature distillation transfers knowledge from intermediate layers of the
teacher model to corresponding layers of the student model. If layer
names are omitted, the final model outputs are used.

## Usage

``` r
distillation_feature(
  loss = "mse",
  teacher_layer_name = NULL,
  student_layer_name = NULL
)
```

## Arguments

- loss:

  Loss function used for feature distillation. This can be a string
  identifier, a Keras loss instance, or a nested list matching the
  layer-output structure. Use `NULL` within a list to skip an output.

- teacher_layer_name:

  Optional name of the teacher layer from which to extract features. The
  final output is used when `NULL`.

- student_layer_name:

  Optional name of the student layer from which to extract features. The
  final output is used when `NULL`.

## Value

A `FeatureDistillation` instance.

## Examples

    loss <- distillation_feature(
      loss = "mse",
      teacher_layer_name = "teacher_features",
      student_layer_name = "student_features"
    )

## See also

Other distillation:  
[`distillation_logits()`](https://keras3.posit.co/dev/reference/distillation_logits.md)  
[`distillation_loss()`](https://keras3.posit.co/dev/reference/distillation_loss.md)  
[`distiller()`](https://keras3.posit.co/dev/reference/distiller.md)  
