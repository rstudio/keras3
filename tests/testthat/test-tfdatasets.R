context("tfdatasets")

test_succeeds("Use tfdatasets to train a keras model", {
  
  model <- keras_model_sequential() %>% 
    layer_dense(units = 1, input_shape = 1)
  model %>% compile(loss='mse', optimizer='sgd')
  
  dataset <- tfdatasets::tensors_dataset(reticulate::tuple(list(1), list(1))) %>% 
    tfdatasets::dataset_repeat(100) %>% 
    tfdatasets::dataset_shuffle(buffer_size = 100) %>% 
    tfdatasets::dataset_batch(10)
  
  if (tensorflow::tf_version() >= "2.0") {
    model %>% fit(dataset, epochs = 2)
    evaluate(model, dataset)
    preds <- predict(model, dataset)
  } else {
    model %>% fit(dataset, epochs = 2, steps_per_epoch = 5)
    evaluate(model, dataset, steps = 10)
    preds <- predict(model, dataset, steps = 10)
  }

})
