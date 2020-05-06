
context("model")

source("utils.R")

test_succeeds("sequential models can be defined", {
  define_model()
})


test_succeeds("sequential models can be compiled", {
  define_and_compile_model()
})

test_succeeds(required_version = "2.0.7", "models can be cloned", {
  model <- define_model()
  model2 <- clone_model(model)
})



# generate dummy training data
data <- matrix(rexp(1000*784), nrow = 1000, ncol = 784)
labels <- matrix(round(runif(1000*10, min = 0, max = 9)), nrow = 1000, ncol = 10)

# genereate dummy input data
input <- matrix(rexp(10*784), nrow = 10, ncol = 784)


test_succeeds("models can be fit, evaluated, and used for predictions", {
  model <- define_and_compile_model()
  fit(model, data, labels, verbose = 0)
  evaluate(model, data, labels)
  predict(model, input)
  predict_on_batch(model, input)
  predict_proba(model, input)
  predict_classes(model, input)
})

test_succeeds("evaluate function returns a named list", {
  model <- define_and_compile_model()
  fit(model, data, labels)
  result <- evaluate(model, data, labels)
  expect_true(!is.null(names(result)))
})

test_succeeds("models can be tested and trained on batches", {
  model <- define_and_compile_model()
  train_on_batch(model, data, labels)
  test_on_batch(model, data, labels)
})


test_succeeds("models layers can be retrieved by name and index", {
  model <- keras_model_sequential() 
  model %>%
    layer_dense(32, input_shape = 784, kernel_initializer = initializer_ones()) %>%
    layer_activation('relu', name = 'first_activation') %>%
    layer_dense(10) %>%
    layer_activation('softmax')
  
  get_layer(model, name = 'first_activation')
  get_layer(model, index = 1)
})


test_succeeds("models layers can be popped", {
  model <- keras_model_sequential() 
  model %>%
    layer_dense(32, input_shape = 784, kernel_initializer = initializer_ones()) %>%
    layer_activation('relu', name = 'first_activation') %>%
    layer_dense(10) %>%
    layer_activation('softmax')
  
  expect_equal(length(model$layers), 4)
  pop_layer(model)
  expect_equal(length(model$layers), 3)
  
})

test_succeeds("can call model with R objects", {
  
  if (!tensorflow::tf_version() >= "1.14") skip("Needs TF >= 1.14")
  
  model <- keras_model_sequential() %>% 
    layer_dense(units = 1, input_shape = 1)
  
  model(
    tensorflow::tf$convert_to_tensor(
      matrix(runif(10), ncol = 1), 
      dtype = tensorflow::tf$float32
    )
  )
  
  input1 <- layer_input(shape = 1)
  input2 <- layer_input(shape = 1)
  
  output <- layer_concatenate(list(input1, input2))
  
  model <- keras_model(list(input1, input2), output)
  l <- lapply(
    list(matrix(runif(10), ncol = 1), matrix(runif(10), ncol = 1)), 
    function(x) tensorflow::tf$convert_to_tensor(x, dtype = tensorflow::tf$float32)
  )
  model(l)
})

test_succeeds("can call a model with additional arguments", {
  
  if (tensorflow::tf_version() < "2.0") skip("needs TF > 2")
  
  model <- keras_model_sequential() %>% 
    layer_dropout(rate = 0.99999999)
  expect_equivalent(as.numeric(model(1, training = TRUE)), 0)
  expect_equivalent(as.numeric(model(1, training = FALSE)), 1)
  
})

test_succeeds("pass validation_data to model fit", {
  
  model <- keras_model_sequential() %>% 
    layer_dense(units =1, input_shape = 2)
  
  model %>% compile(loss = "mse", optimizer = "sgd")
  
  model %>% 
    fit(
      matrix(runif(100), ncol = 2), y = runif(50),
      batch_size = 10,
      validation_data = list(matrix(runif(100), ncol = 2), runif(50))
    )
  
})





