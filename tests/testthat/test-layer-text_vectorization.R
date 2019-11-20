context("layer_text_vectorization")

source("utils.R")

test_call_succeeds("layer_text_vectorization", {
  
  if (tensorflow::tf_version() < "2.1")
    skip("TextVectorization requires TF version >= 2.1")
  
  input <- matrix(c("hello world", "hello world"), ncol = 1)
  
  layer <- layer_text_vectorization()
  layer %>% adapt(input)
  output <- layer(input)

  expect_s3_class(output, "tensorflow.tensor")
})

test_call_succeeds("layer_text_vectorization", {
  
  if (tensorflow::tf_version() < "2.1")
    skip("TextVectorization requires TF version >= 2.1")
  
  x <- matrix(c("hello world", "hello world"), ncol = 1)
  
  layer <- layer_text_vectorization(output_mode = "binary", 
                                    pad_to_max_tokens = FALSE)
  layer %>% adapt(x)
  
  output <- layer(x)
  
  expect_s3_class(output, "tensorflow.tensor")
})

test_call_succeeds("can use layer_text_vectorization in a functional model", {
  
  if (tensorflow::tf_version() < "2.1")
    skip("TextVectorization requires TF version >= 2.1")
  
  x <- matrix(c("hello world", "hello world"), ncol = 1)
  
  layer <- layer_text_vectorization()
  layer %>% adapt(x)
  
  input <- layer_input(shape = 1, dtype = "string")
  output <- layer(input)
  model <- keras_model(input, output)
  
  pred <- predict(model, x)
  
})

test_call_succeeds("can set and get the vocabulary of layer_text_vectorization", {
  
  if (tensorflow::tf_version() < "2.1")
    skip("TextVectorization requires TF version >= 2.1")
  
  x <- matrix(c("hello world", "hello world"), ncol = 1)
  
  layer <- layer_text_vectorization()
  set_vocabulary(layer, vocab = c("hello", "world"))
  
  output <- layer(x)
  
  vocab <- get_vocabulary(layer)
  
  expect_s3_class(output, "tensorflow.tensor")
  expect_length(vocab, 2)
})


test_call_succeeds("can use layer_text_vectorization", {
  if (tensorflow::tf_version() < "2.1")
    skip("TextVectorization requires TF version >= 2.1")
  
  x <- matrix(c("hello world", "hello world"), ncol = 1)
  x_ds <- tfdatasets::tensor_slices_dataset(x)
  
  layer <- layer_text_vectorization()
  layer %>% adapt(x_ds)
  expect_length(get_vocabulary(layer), 2)
})



