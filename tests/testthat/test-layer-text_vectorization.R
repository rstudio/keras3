context("layer_text_vectorization")



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
  layer$get_vocabulary()
  set_vocabulary(layer, vocab = c("hello", "world"))
  
  output <- layer(x)
  
  vocab <- get_vocabulary(layer)
  
  expect_s3_class(output, "tensorflow.tensor")
  if (tensorflow::tf_version() < "2.3")
    expect_length(vocab, 2)
  else
    expect_length(vocab, 4) # 0 is used for padding and 1 for unknown.
})


test_call_succeeds("can use layer_text_vectorization", {
  if (tensorflow::tf_version() < "2.1")
    skip("TextVectorization requires TF version >= 2.1")
  
  x <- matrix(c("hello world", "hello world"), ncol = 1)
  x_ds <- tfdatasets::tensor_slices_dataset(x)
  
  layer <- layer_text_vectorization()
  layer %>% adapt(x_ds)
  
  if (tensorflow::tf_version() < "2.3")
    expect_length(get_vocabulary(layer), 2)
  else
    expect_length(get_vocabulary(layer), 4) # 0 is used for padding and 1 for unknown.
})


test_call_succeeds("can create a tf-idf layer", {
  
  if (tensorflow::tf_version() < "2.1")
    skip("TextVectorization requires TF version >= 2.1")
  
  num_words <- 10000 
  max_length <- 50 
  
  text_vectorization <- layer_text_vectorization( 
    max_tokens = num_words, output_mode = "tf-idf" 
  )
  text_vectorization %>% adapt(c("hello world", "hello"))
  x <- text_vectorization(matrix(c("hello"), ncol = 1))
  
  expect_s3_class(x, "tensorflow.tensor")
  
})


