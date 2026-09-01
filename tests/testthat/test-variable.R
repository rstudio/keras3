test_that("keras_variable exposes the layout argument", {
  expect_true("layout" %in% names(formals(keras_variable)))
  expect_null(formals(keras_variable)$layout)
})
