test_that("model quantization accepts configs and filters", {
  skip_if_no_keras("3.15.1")

  model <- keras_model_sequential(input_shape = 2L) |>
    layer_dense(2L, name = "keep") |>
    layer_dense(1L, name = "skip")
  invisible(model(op_ones(c(1, 2))))

  result <- withVisible(quantize_weights(
    model,
    config = quantizer_int8_quantization_config(),
    filters = "keep"
  ))

  expect_false(result$visible)
  expect_identical(result$value, model)
  expect_identical(
    model$get_layer("keep")$dtype_policy$name,
    "int8_from_float32"
  )
  expect_identical(model$get_layer("skip")$dtype_policy$name, "float32")
})
