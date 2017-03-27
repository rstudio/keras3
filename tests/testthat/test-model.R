
context("model")

source("utils.R")

# generate dummy training data
data <- matrix(rexp(1000*784), nrow = 1000, ncol = 784)
labels <- matrix(round(runif(1000*10, min = 0, max = 9)), nrow = 1000, ncol = 10)

# genereate dummy input data
input <- matrix(rexp(10*784), nrow = 10, ncol = 784)

define_model <- function() {
  model_sequential() %>%
    layer_dense(32, input_shape = 784) %>%
    layer_activation('relu') %>%
    layer_dense(10) %>%
    layer_activation('softmax')
}

define_and_compile_model <- function() {
  define_model() %>% 
    compile(
      loss='binary_crossentropy',
      optimizer = optimizer_sgd(),
      metrics='accuracy'
    )
}


test_succeeds("sequential models can be defined", {
  define_model()
})


test_succeeds("sequential models can be compiled", {
  define_and_compile_model()
})

test_succeeds("models can be fit, evaluated, and used for predictions", {
  model <- define_and_compile_model()
  fit(model, data, labels)
  evaluate(model, data, labels)
  predict(model, input)
})


test_succeeds("model can be saved and loaded", {
  
  if (!keras:::have_h5py())
    skip("h5py not available for testing")
  
  model <- define_and_compile_model()
  tmp <- tempfile("model", fileext = ".hdf5")
  write_model(model, tmp)
  model <- read_model(tmp)
})

