context("preprocessing")

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

test_call_succeeds("missing text data", required_version = "2.0.5", {
  expect_error(text_hashing_trick(letters, 10),
               "`text` should be length 1")

  texts <- c(
    'Dogs and cats living together.',
    NA_character_
  )
  expect_true(all(!is.na(text_hashing_trick(texts[1], 10))))
  expect_true(all( is.na(text_hashing_trick(texts[2], 10))))
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
    img_arr1 <- imagenet_preprocess_input(img_arr)
    img_arr2 <- preprocess_input(img_arr, tensorflow::tf$keras$applications$imagenet_utils$preprocess_input)
    expect_equal(img_arr1, img_arr2)
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
      y_col = "class"
    )

    batch <- reticulate::iter_next(img_gen)

    expect_equal(dim(batch[[1]]), c(10, 256, 256, 3))
    expect_equal(dim(batch[[2]]), c(10, 10))

    if (tensorflow::tf_version() >= "2.3") {

      expect_warning(
        img_gen <- flow_images_from_dataframe(
          df,
          directory = ".",
          x_col = "fname",
          y_col = "class",
          drop_duplicates = TRUE
        )
      )

    }

  }
})

test_succeeds("flow images from directory works", {

  if (!have_pillow())
    skip("Pillow required.")

  dir <- tempfile()
  dir.create(dir)
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

  expect_warning_if(tensorflow::tf_version() >= "2.1", {
    # test fitting the model
    model %>% fit_generator(gen, steps_per_epoch = 20)

    # test predictions
    preds <- predict_generator(model, gen, steps = 10)

    # evaluate
    eva <- evaluate_generator(model, gen, steps = 10)
  })
})

test_succeeds("images_dataset_from_directory", {

  if (tensorflow::tf_version() < "2.3")
    skip("requires tf_version() >= 2.3")

  dir <- tempfile()
  dir.create(dir)
  dir.create(file.path(dir, "0"))
  dir.create(file.path(dir, "1"))

  mnist <- dataset_mnist()
  ind <- which(mnist$train$y %in% c(0, 1))

  for (i in ind) {
    img <- mnist$train$x[i,,]/255
    rname <- paste(sample(letters, size = 10, replace = TRUE), collapse = "")
    jpeg::writeJPEG(img, target = paste0(dir, "/", mnist$train$y[i], "/", rname, ".jpeg"))
  }

  data <- image_dataset_from_directory(dir)

  iter <- reticulate::as_iterator(data)
  d <- reticulate::iter_next(iter)

  expect_equal(d[[1]]$shape$as_list(), c(32, 256, 256, 3))
  expect_equal(d[[2]]$shape$as_list(), c(32))

})

if(tf_version() >= "2.6")
test_succeeds("timeseries_dataset_from_array", {

  # example 1 in docs
  dsi <- timeseries_dataset_from_array(
    0:100, NULL, sequence_length = 10, sampling_rate = 2, batch_size = 11,
    sequence_stride = 3, shuffle = FALSE)$as_numpy_iterator()

  batches <- reticulate::iterate(dsi, simplify = FALSE)

    # generated with dput()
    expected <- list(structure(c(0L, 3L, 6L, 9L, 12L, 15L, 18L, 21L, 24L, 27L,
  30L, 2L, 5L, 8L, 11L, 14L, 17L, 20L, 23L, 26L, 29L, 32L, 4L,
  7L, 10L, 13L, 16L, 19L, 22L, 25L, 28L, 31L, 34L, 6L, 9L, 12L,
  15L, 18L, 21L, 24L, 27L, 30L, 33L, 36L, 8L, 11L, 14L, 17L, 20L,
  23L, 26L, 29L, 32L, 35L, 38L, 10L, 13L, 16L, 19L, 22L, 25L, 28L,
  31L, 34L, 37L, 40L, 12L, 15L, 18L, 21L, 24L, 27L, 30L, 33L, 36L,
  39L, 42L, 14L, 17L, 20L, 23L, 26L, 29L, 32L, 35L, 38L, 41L, 44L,
  16L, 19L, 22L, 25L, 28L, 31L, 34L, 37L, 40L, 43L, 46L, 18L, 21L,
  24L, 27L, 30L, 33L, 36L, 39L, 42L, 45L, 48L), .Dim = 11:10),
      structure(c(33L, 36L, 39L, 42L, 45L, 48L, 51L, 54L, 57L,
      60L, 63L, 35L, 38L, 41L, 44L, 47L, 50L, 53L, 56L, 59L, 62L,
      65L, 37L, 40L, 43L, 46L, 49L, 52L, 55L, 58L, 61L, 64L, 67L,
      39L, 42L, 45L, 48L, 51L, 54L, 57L, 60L, 63L, 66L, 69L, 41L,
      44L, 47L, 50L, 53L, 56L, 59L, 62L, 65L, 68L, 71L, 43L, 46L,
      49L, 52L, 55L, 58L, 61L, 64L, 67L, 70L, 73L, 45L, 48L, 51L,
      54L, 57L, 60L, 63L, 66L, 69L, 72L, 75L, 47L, 50L, 53L, 56L,
      59L, 62L, 65L, 68L, 71L, 74L, 77L, 49L, 52L, 55L, 58L, 61L,
      64L, 67L, 70L, 73L, 76L, 79L, 51L, 54L, 57L, 60L, 63L, 66L,
      69L, 72L, 75L, 78L, 81L), .Dim = 11:10), structure(c(66L,
      69L, 72L, 75L, 78L, 81L, 68L, 71L, 74L, 77L, 80L, 83L, 70L,
      73L, 76L, 79L, 82L, 85L, 72L, 75L, 78L, 81L, 84L, 87L, 74L,
      77L, 80L, 83L, 86L, 89L, 76L, 79L, 82L, 85L, 88L, 91L, 78L,
      81L, 84L, 87L, 90L, 93L, 80L, 83L, 86L, 89L, 92L, 95L, 82L,
      85L, 88L, 91L, 94L, 97L, 84L, 87L, 90L, 93L, 96L, 99L), .Dim = c(6L,
      10L)))

  expect_equal(batches, expected)


  # example 2 in docs
  steps <- 100
  # data is integer seq with some noise
  data <- array(1:steps + abs(rnorm(steps, sd = .25)))
  inputs_data <- head(data, -10) # drop last 10
  targets <- tail(data, -10)    # drop first 10
  dataset <- timeseries_dataset_from_array(
    inputs_data, targets, sequence_length=10)

  dataset_iterator <- reticulate::as_iterator(dataset)
  repeat {
    batch <- reticulate::iter_next(dataset_iterator)
    if(is.null(batch)) break
    c(input, target) %<-% batch
    stopifnot(exprs = {
      # First sequence: steps [1-10]
      # Corresponding target: step 11
      all.equal(as.array(input[1, ]), data[1:10])
      all.equal(as.array(target[1]), data[11])

      all.equal(as.array(input[2, ]), data[2:11])
      all.equal(as.array(target[2]), data[12])

      all.equal(as.array(input[3, ]), data[3:12])
      all.equal(as.array(target[3]), data[13])
    })
  }


  # example 3 from docs
  X <- seq(100)
  Y <- X * 2

  sample_length <- 20
  input_dataset <- timeseries_dataset_from_array(X,
                                                 NULL,
                                                 sequence_length = sample_length,
                                                 sequence_stride = sample_length)
  target_dataset <- timeseries_dataset_from_array(Y,
                                                  NULL,
                                                  sequence_length = sample_length,
                                                  sequence_stride = sample_length)

  dataset_iterator <-
    tfdatasets::zip_datasets(tuple(input_dataset, target_dataset))$as_numpy_iterator()
  while (!is.null(batch <- reticulate::iter_next(dataset_iterator))) {
    c(inputs, targets) %<-% batch
    stopifnot(
      all.equal(inputs[1,], X[1:sample_length]),
      all.equal(targets[1,], Y[1:sample_length]),
      # second sample equals output timestamps 20-40
      all.equal(inputs[2,], X[(1:sample_length) + sample_length]),
      all.equal(targets[2,], Y[(1:sample_length) + sample_length])
    )
  }
})
