context("optimizers")

test_that("Muon preserves fractional Adam learning-rate ratios", {
  skip_if_no_keras("3.14.0")

  optimizer <- optimizer_muon(adam_lr_ratio = 0.5)

  expect_equal(optimizer$adam_lr_ratio, 0.5)
})


test_optimizer <- function(name) {
  optimizer_fn <- eval(parse(text = name))
  test_call_succeeds(name, {
    keras_model_sequential() %>%
      layer_dense(32, input_shape = c(784)) %>%
      compile(
        optimizer = optimizer_fn(),
        loss='binary_crossentropy',
        metrics='accuracy'
      )
  })
}


test_optimizer("optimizer_sgd")
test_optimizer("optimizer_rmsprop")
test_optimizer("optimizer_adagrad")
test_optimizer("optimizer_adadelta")
test_optimizer("optimizer_adam")
test_optimizer("optimizer_adamax")
test_optimizer("optimizer_nadam")
