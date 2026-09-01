context("optimizers")

test_that("Muon preserves fractional Adam learning-rate ratios", {
  skip_if_no_keras("3.14.0")

  optimizer <- optimizer_muon(adam_lr_ratio = 0.5)

  expect_equal(optimizer$adam_lr_ratio, 0.5)
})

test_that("optimizer names default to automatic naming", {
  optimizer_constructors <- list(
    optimizer_adadelta = optimizer_adadelta,
    optimizer_adafactor = optimizer_adafactor,
    optimizer_adagrad = optimizer_adagrad,
    optimizer_adam = optimizer_adam,
    optimizer_adam_w = optimizer_adam_w,
    optimizer_adamax = optimizer_adamax,
    optimizer_ftrl = optimizer_ftrl,
    optimizer_lamb = optimizer_lamb,
    optimizer_lion = optimizer_lion,
    optimizer_muon = optimizer_muon,
    optimizer_nadam = optimizer_nadam,
    optimizer_rmsprop = optimizer_rmsprop,
    optimizer_sgd = optimizer_sgd
  )

  for (optimizer_name in names(optimizer_constructors)) {
    optimizer_constructor <- optimizer_constructors[[optimizer_name]]
    expect_null(formals(optimizer_constructor)$name, info = optimizer_name)
  }
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
