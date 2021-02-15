context("history")


test_succeeds("as.data.frame works for history with early stopping", {
  
  early_stop <- callback_early_stopping(monitor = "loss", patience = 1)
  
  model <- keras_model_sequential() %>% 
    layer_dense(1, input_shape = 1)
  
  model %>% compile(loss = "mse", optimizer = "adam")
  
  x = matrix(runif(100), ncol = 1)
  
  history <- model %>% fit(
    x = x,
    y = x[,1] + rnorm(100, 0.1),
    epochs = 500,
    validation_split = 0.2,
    verbose = 0,
    callbacks = list(early_stop)
  )
  
  expect_error(
    d <- as.data.frame(history),
    regexp = NA
  )
  
  expect_error(
    plot(history),
    regexp = NA
  )
  
})
