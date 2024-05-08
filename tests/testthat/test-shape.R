test_that("shape() works", {

  expect_identical(shape(NULL, 28),
                   structure(list(NULL, 28L), class = "keras_shape"))

})
