

context("layer methods")

source("utils.R")


test_succeeds("model can be saved and loaded from config", {
  
  inputs <- layer_input(shape = c(784))
  predictions <- inputs %>%
    layer_dense(units = 64, activation = 'relu') %>%
    layer_dense(units = 64, activation = 'relu') %>%
    layer_dense(units = 10, activation = 'softmax')
  model <- keras_model(inputs = inputs, outputs = predictions)
  
  config <- get_config(model)
  model_from <- from_config(config)
})

test_succeeds("sequential model can be saved and loaded from config", {
  model <- define_model()
  config <- get_config(model)
  model_from <- from_config(config)
})

test_succeeds("layer can saved and loaded from config", {
  layer <- layer_dense(units = 64)
  config <- get_config(layer)
  layer_from <- from_config(config)
})


test_succeeds("model weights as R array can be read and written", {
  model <- define_and_compile_model()
  weights <- get_weights(model)
  set_weights(model, weights)
})


# generate dummy training data
data <- matrix(rexp(1000*784), nrow = 1000, ncol = 784)
labels <- matrix(round(runif(1000*10, min = 0, max = 9)), nrow = 1000, ncol = 10)

# genereate dummy input data
input <- matrix(rexp(10*784), nrow = 10, ncol = 784)


test_succeeds("layer weights as R array can be read and written", {
  model <- define_and_compile_model()
  fit(model, data, labels)
  
  layer <- model$layers[[1]]
  weights <- get_weights(layer)
  set_weights(layer, weights)
})

test_succeeds("model parameters can be counted", {
  model <- define_and_compile_model()
  count_params(model)
})

test_succeeds("layer parameters can be counted", {
  model <- define_and_compile_model()
  layer <- model$layers[[1]]
  count_params(layer)
})


test_succeeds("layer node functions are accessible", {
  model <- define_model()
  layer <- model$layers[[2]]
  get_input_at(layer, 1)
  get_output_at(layer, 1)
  get_input_shape_at(layer, 1)
  get_output_shape_at(layer, 1)
  get_input_mask_at(layer, 1)
  get_output_mask_at(layer, 1)
})


test_succeeds("layer state can be reset", {
  skip_if_cntk() # CNTK backend does not support stateful RNNs, see
                 # https://docs.microsoft.com/en-us/cognitive-toolkit/using-cntk-with-keras#known-issues
  model <- keras_model_sequential()
  model %>% 
    layer_lstm(units = 32, input_shape=c(10, 16), batch_size=32, stateful=TRUE) %>% 
    layer_dense(units = 16, activation = 'softmax')
  
  layer <- model$layers[[1]]
  reset_states(layer)
})


