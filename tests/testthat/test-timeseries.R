context("timeseries")

source("utils.R")

test_call_succeeds("timeseries_generator", required_version = "2.1.5", {

  data <- matrix(1:50, nrow = 50, ncol = 1)
  targets <- matrix(1:50, nrow = 50, ncol = 1)
  data_gen <- timeseries_generator(data, targets, length = 10, sampling_rate = 2, batch_size = 2)
  
  model <- keras_model_sequential() %>% 
    layer_lstm(units = 5, input_shape = c(5, 1)) %>%
    layer_dense(units = 1)
    
  model %>% compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = "accuracy"
  )
  
  model %>% fit_generator(data_gen, steps_per_epoch = 10)
})
