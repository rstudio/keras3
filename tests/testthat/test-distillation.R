test_that("concrete distillation loss constructors expose the public API", {
  skip_if_no_keras("3.15.1")

  expect_true(is.function(distillation_feature))
  expect_true(is.function(distillation_logits))

  expect_s3_class(
    distillation_feature(loss = "mse"),
    "keras.src.distillation.distillation_loss.FeatureDistillation"
  )
  expect_s3_class(
    distillation_logits(temperature = 2),
    "keras.src.distillation.distillation_loss.LogitsDistillation"
  )
})

test_that("distiller trains with a concrete distillation loss", {
  skip_if_no_keras("3.15.1")

  expect_true(is.function(distiller))
  expect_named(
    formals(distiller),
    c(
      "teacher", "student", "distillation_losses",
      "distillation_loss_weights", "student_loss_weight", "name", "..."
    )
  )

  teacher <- keras_model_sequential(input_shape = 2) |>
    layer_dense(1)
  student <- keras_model_sequential(input_shape = 2) |>
    layer_dense(1)

  model <- distiller(
    teacher = teacher,
    student = student,
    distillation_losses = distillation_logits()
  )

  expect_s3_class(model, "keras.src.distillation.distiller.Distiller")

  model |> compile(optimizer = "adam", loss = "mse")
  x <- matrix(seq_len(8) / 8, ncol = 2)
  y <- matrix(rowSums(x), ncol = 1)
  history <- model |> fit(x, y, epochs = 1, verbose = 0)

  expect_true(all(
    c("distillation_loss", "student_loss", "total_loss") %in%
      names(history$metrics)
  ))
})
