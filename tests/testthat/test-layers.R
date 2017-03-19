context("layers")

source("utils.R")


test_that("layer_dense call succeeds", {
  skip_if_no_keras()
  layer_dense(model_sequential(), 32, input_dim = 784)
  expect_equal(TRUE, TRUE)
})

