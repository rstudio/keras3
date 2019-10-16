context("layer_text_vectorization")

source("utils.R")

test_call_succeeds("layer_text_vectorization", {
  
  if (tensorflow::tf_version() < "2.1")
    skip("TextVectorization requires TF version >= 2.1")
  
  input <- matrix(c("hello world", "hello world"), ncol = 1)
  
  layer <- layer_text_vectorization()
  layer$adapt(input)
  output <- layer(input)

  expect_s3_class(output, "tensorflow.tensor")
})

test_call_succeeds("can use layer_text_vectorization in a functional model", {
  
  if (tensorflow::tf_version() < "2.1")
    skip("TextVectorization requires TF version >= 2.1")
  
  x <- matrix(c("hello world", "hello world"), ncol = 1)
  
  layer <- layer_text_vectorization()
  layer$adapt(x)
  
  input <- layer_input(shape = 1, dtype = "string")
  output <- layer(input)
  model <- keras_model(input, output)
  
  pred <- predict(model, x)
  
})



