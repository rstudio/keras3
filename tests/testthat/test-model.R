
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
  fit(model, data, labels)
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





