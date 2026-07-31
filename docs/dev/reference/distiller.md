# Model for transferring knowledge from a teacher to a student

A distiller trains a student model from both ground-truth labels and the
predictions or intermediate features of a frozen teacher model.

## Usage

``` r
distiller(
  teacher,
  student,
  distillation_losses,
  distillation_loss_weights = NULL,
  student_loss_weight = 0.5,
  name = "distiller",
  ...
)
```

## Arguments

- teacher:

  Trained Keras model that provides the knowledge to transfer. The
  teacher is frozen by the distiller.

- student:

  Keras model to train.

- distillation_losses:

  A distillation loss or list of distillation losses.

- distillation_loss_weights:

  Optional numeric vector of weights, one for each element of
  `distillation_losses`.

- student_loss_weight:

  Weight of the student's supervised loss. Must be between 0 and 1.

- name:

  Name of the distiller model.

- ...:

  Additional arguments passed to `keras.Model`.

## Value

A Keras `Distiller` model.

## Examples

    teacher <- keras_model_sequential(input_shape = 4) |>
      layer_dense(8, activation = "relu") |>
      layer_dense(3)
    student <- keras_model_sequential(input_shape = 4) |>
      layer_dense(3)

    model <- distiller(
      teacher = teacher,
      student = student,
      distillation_losses = distillation_logits(temperature = 3)
    )
    model |> compile(optimizer = "adam", loss = "mse")

## See also

Other distillation:  
[`distillation_feature()`](https://keras3.posit.co/dev/reference/distillation_feature.md)  
[`distillation_logits()`](https://keras3.posit.co/dev/reference/distillation_logits.md)  
[`distillation_loss()`](https://keras3.posit.co/dev/reference/distillation_loss.md)  

Other model creation:  
[`keras_input()`](https://keras3.posit.co/dev/reference/keras_input.md)  
[`keras_model()`](https://keras3.posit.co/dev/reference/keras_model.md)  
[`keras_model_sequential()`](https://keras3.posit.co/dev/reference/keras_model_sequential.md)  
