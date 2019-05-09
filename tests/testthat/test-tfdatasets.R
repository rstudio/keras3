context("tfdatasets")

test_succeeds("Use an R-based custom Keras model", {
  
  model <- keras_model_sequential() %>% 
    layer_dense(units = 1, input_shape = 1)
  model %>% compile(loss='mse', optimizer='sgd')
  
  dataset <- tfdatasets::tensors_dataset(reticulate::tuple(list(1), list(1))) %>% 
    tfdatasets::dataset_repeat(100) %>% 
    tfdatasets::dataset_shuffle(buffer_size = 100) %>% 
    tfdatasets::dataset_batch(10)
  
  model %>% fit(dataset, epochs = 2)
  
  evaluate(model, dataset)
  
  preds <- predict(model, dataset)
  
})
