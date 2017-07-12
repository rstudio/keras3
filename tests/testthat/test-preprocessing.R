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
  
  for (mode in c('binary', 'count', 'tfidf', 'freq'))
    texts_to_matrix(tokenizer, texts, mode)
})

test_succeeds("loading an image for preprocessing", {
  if (have_pillow()) {
    img <- image_load("digit.jpeg")
    img_arr <- image_to_array(img)
  }
})
