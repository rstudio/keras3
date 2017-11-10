context("freeze")

source("utils.R")


define_freeze_model <- function() {
  model <- keras_model_sequential() 
  model %>%
    layer_dense(32, input_shape = 784, kernel_initializer = initializer_ones(), name = "input") %>%
    layer_activation('relu', name = "relu_activation") %>%
    layer_dense(10, name = "dense") %>%
    layer_activation('softmax', name = "softmax_activation")
  model
}

test_succeeds("freeze_weights can freeze an entire model", {
  model <- define_freeze_model()
  freeze_weights(model)
  expect_length(model$trainable_weights, 0)
})

test_succeeds("model can be unfrozen after freezing", {
  model <- define_freeze_model()
  freeze_weights(model)
  unfreeze_weights(model)
  expect_length(model$trainable_weights, 4)
})

test_succeeds("freeze_weights can work on indexes", {
  model <- define_freeze_model()
  freeze_weights(model, from = 2, to = 3)
  expect_length(model$trainable_weights, 2)
})

test_succeeds("freeze_weights can work on names", {
  model <- define_freeze_model()
  freeze_weights(model, from = "dense")
  expect_length(model$trainable_weights, 2)
})


