# Feature distillation loss

Feature distillation transfers knowledge from intermediate layers of the
teacher model to corresponding layers of the student model. This can
help the student learn better internal representations and often leads
to better performance than logits-only distillation. If layer names are
omitted, the final model outputs are used.

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
  identifier such as `"mse"`, `"cosine_similarity"`, or `"mae"`; a Keras
  loss instance; or a nested list of losses matching the layer-output
  structure. Use `NULL` within a list to skip distillation for that
  output. At least one loss must be non-`NULL`. Defaults to `"mse"`.

- teacher_layer_name:

  Name of the teacher layer from which to extract features. The final
  output is used when `NULL`, the default.

- student_layer_name:

  Name of the student layer from which to extract features. The final
  output is used when `NULL`, the default.

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
[`distiller()`](https://keras3.posit.co/dev/reference/distiller.md)  
