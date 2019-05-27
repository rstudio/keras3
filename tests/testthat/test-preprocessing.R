context("preprocessing")

source("utils.R")

test_call_succeeds("pad_sequences", {
  a <- list(list(1), list(1,2), list(1,2,3))
  b <- pad_sequences(a, maxlen=3, padding='pre')    
  expect_equal(b, matrix(c(0L, 0L, 1L, 0L, 1L, 2L, 1L, 2L, 3L), nrow = 3, ncol = 3))
})

test_call_succeeds("make_sampling_table", {
  make_sampling_table(size = 40)
})

test_call_succeeds("skipgrams", {
  skipgrams(1:3, vocabulary_size = 3)
})

test_call_succeeds("text_one_hot", {
  text <- 'The cat sat on the mat.'
  encoded <- text_one_hot(text, 5)
  expect_equal(length(encoded), 6)
})

test_call_succeeds("text_hashing_trick", required_version = "2.0.5", {
  text <- 'The cat sat on the mat.'
  encoded <- text_hashing_trick(text, 5)
  expect_equal(length(encoded), 6)
})

test_succeeds("use of text tokenizer", {
  texts <- c(
    'The cat sat on the mat.',
    'The dog sat on the log.',
    'Dogs and cats living together.'
  )
  tokenizer <- text_tokenizer(num_words = 10)
  tokenizer %>% fit_text_tokenizer(texts)
  
  sequences <- iterate(texts_to_sequences_generator(tokenizer, texts))
  tokenizer %>% fit_text_tokenizer(sequences)
  
  # save and then reload tokenizer
  save_text_tokenizer(tokenizer, "tokenizer")
  on.exit(unlink("tokenizer"), add = TRUE)
  tokenizer <- load_text_tokenizer("tokenizer")
  
  for (mode in c('binary', 'count', 'tfidf', 'freq'))
    texts_to_matrix(tokenizer, texts, mode)
})

test_succeeds("image can be preprocessed", {
  if (have_pillow()) {
    img <- image_load("digit.jpeg")
    img_arr <- image_to_array(img)
    img_arr <- array_reshape(img_arr, c(1, dim(img_arr)))
    img_arr <- imagenet_preprocess_input(img_arr)
  }
})

test_succeeds("images arrays can be saved", {
  if (have_pillow()) {
    img <- image_load("digit.jpeg")
    img_arr <- image_to_array(img)
    image_array_save(img_arr, "digit2.jpeg")
  }
})

test_succeeds("images arrays can be resized", {
  if (have_pillow()) {
    img <- image_load("digit.jpeg")
    img_arr <- image_to_array(img)
    image_array_resize(img_arr, height = 450, width = 448) %>% 
      image_array_save("digit_resized.jpeg")
  }
})

test_succeeds("flow images from dataframe works", {
  
  if (!reticulate::py_module_available("pandas"))
    skip("Needs pandas")
    
  
  if (have_pillow()) {
    
    df <- data.frame(
      fname = rep("digit.jpeg", 10), 
      class = letters[1:10], 
      stringsAsFactors = FALSE
      )
    img_gen <- flow_images_from_dataframe(
      df, 
      directory = ".", 
      x_col = "fname", 
      y_col = "class", 
      drop_duplicates = FALSE
      )
    
    batch <- reticulate::iter_next(img_gen)
    
    expect_equal(dim(batch[[1]]), c(10, 256, 256, 3))
    expect_equal(dim(batch[[2]]), c(10, 10))
  }
})

test_succeeds("flow images from directory works", {
  
  if (!have_pillow())
    skip("Pillow required.")
  
  dir <- tempdir()
  dir.create(paste0(dir, "/flow-img"))
  dir <- paste0(dir, "/flow-img")
  dir.create(paste0(dir, "/0"))
  dir.create(paste0(dir, "/1"))
  
  mnist <- dataset_mnist()
  ind <- which(mnist$train$y %in% c(0, 1))
  
  for (i in ind) {
    img <- mnist$train$x[i,,]/255
    rname <- paste(sample(letters, size = 10, replace = TRUE), collapse = "")
    jpeg::writeJPEG(img, target = paste0(dir, "/", mnist$train$y[i], "/", rname, ".jpeg"))
  }
  
  img_gen <- image_data_generator()
  
  gen <- flow_images_from_directory(
    directory = dir, 
    generator = img_gen, 
    target_size = c(28, 28), 
    batch_size = 32
  )
  
  model <- keras_model_sequential() %>% 
    layer_flatten(input_shape = c(28,28, 3)) %>% 
    layer_dense(units = 2, activation = "softmax")
  
  model %>% compile(loss = "binary_crossentropy", optimizer = "adam")
  
  # test fitting the model
  model %>% fit_generator(gen, steps_per_epoch = 20)
  
  # test predictions
  preds <- predict_generator(model, gen, steps = 10)
  
  # evaluate
  eva <- evaluate_generator(model, gen, steps = 10)
})


