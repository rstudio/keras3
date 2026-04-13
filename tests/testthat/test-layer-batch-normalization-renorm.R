test_that("batch normalization preserves renorm configuration", {
  skip_if_no_keras("3.14.0")

  config <- layer_batch_normalization(
    renorm = TRUE,
    renorm_clipping = list(rmax = 2, rmin = 0.5, dmax = 1),
    renorm_momentum = 0.8
  )$get_config()

  expect_true(config$renorm)
  expect_equal(
    config$renorm_clipping,
    list(rmax = 2, rmin = 0.5, dmax = 1)
  )
  expect_equal(config$renorm_momentum, 0.8)
})
