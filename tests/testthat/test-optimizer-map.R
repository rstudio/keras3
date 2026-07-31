test_that("optimizer maps dispatch variables to optimizers", {
  optimizers <- optimizer_map(
    default_optimizer = optimizer_sgd(name = "sgd"),
    optimizer_map = list("encoder/.*" = optimizer_adam(name = "adam"))
  )

  expect_equal(optimizers[["encoder/dense"]]$name, "adam")
  expect_equal(optimizers[["decoder/dense"]]$name, "sgd")

  optimizer <- optimizer_multi(optimizers)
  expect_s3_class(optimizer, "keras.src.optimizers.multi_optimizer.MultiOptimizer")
})
