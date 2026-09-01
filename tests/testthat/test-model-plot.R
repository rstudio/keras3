test_that("model plots forward spline configuration", {
  skip_if_no_keras("3.14.0")
  skip_if_not(reticulate::py_module_available("pydot"))

  model <- keras_model_sequential(input_shape = 2L) |>
    layer_dense(1L)
  path <- tempfile(fileext = ".raw")
  on.exit(unlink(path))

  plot(model, to_file = path, splines = "curved")

  expect_true(any(grepl("splines=curved", readLines(path), fixed = TRUE)))
})
