#' This example demonstrates the use of fasttext for text classification
#' 
#' Based on Joulin et al's paper: 
#' "Bags of Tricks for Efficient Text Classification"
#' https://arxiv.org/abs/1607.01759
#' 
#' Results on IMDB datasets with uni and bi-gram embeddings:
#'  Uni-gram: 0.8813 test accuracy after 5 epochs. 8s/epoch on i7 CPU
#'  Bi-gram : 0.9056 test accuracy after 5 epochs. 2s/epoch on GTx 980M GPU
#' 

library(keras)
library(purrr)

# Function Definitions ----------------------------------------------------

create_ngram_set <- function(input_list, ngram_value = 2){
  indices <- map(0:(length(input_list) - ngram_value), ~1:ngram_value + .x)
  indices %>%
    map_chr(~input_list[.x] %>% paste(collapse = "|")) %>%
    unique()
}

add_ngram <- function(sequences, token_indice, ngram_range = 2){
  ngrams <- map(
    sequences, 
    create_ngram_set, ngram_value = ngram_range
  )
  
  seqs <- map2(sequences, ngrams, function(x, y){
    tokens <- token_indice$token[token_indice$ngrams %in% y]  
    c(x, tokens)
  })
  
  seqs
}


# Parameters --------------------------------------------------------------

# ngram_range = 2 will add bi-grams features
ngram_range <- 2
max_features <- 20000
maxlen <- 400
batch_size <- 32
embedding_dims <- 50
epochs <- 5


# Data Preparation --------------------------------------------------------

# Load data
imdb_data <- dataset_imdb(num_words = max_features)

# Train sequences
print(length(imdb_data$train$x))
print(sprintf("Average train sequence length: %f", mean(map_int(imdb_data$train$x, length))))

# Test sequences
print(length(imdb_data$test$x)) 
print(sprintf("Average test sequence length: %f", mean(map_int(imdb_data$test$x, length))))

if(ngram_range > 1) {
  
  # Create set of unique n-gram from the training set.
  ngrams <- imdb_data$train$x %>% 
    map(create_ngram_set) %>%
    unlist() %>%
    unique()

  # Dictionary mapping n-gram token to a unique integer
    # Integer values are greater than max_features in order
    # to avoid collision with existing features
  token_indice <- data.frame(
    ngrams = ngrams,
    token  = 1:length(ngrams) + (max_features), 
    stringsAsFactors = FALSE
  )
  
  # max_features is the highest integer that could be found in the dataset
  max_features <- max(token_indice$token) + 1
  
  # Augmenting x_train and x_test with n-grams features
  imdb_data$train$x <- add_ngram(imdb_data$train$x, token_indice, ngram_range)
  imdb_data$test$x <- add_ngram(imdb_data$test$x, token_indice, ngram_range)
}

# Pad sequences
imdb_data$train$x <- pad_sequences(imdb_data$train$x, maxlen = maxlen)
imdb_data$test$x <- pad_sequences(imdb_data$test$x, maxlen = maxlen)


# Model Definition --------------------------------------------------------

model <- keras_model_sequential()

model %>%
  layer_embedding(
    input_dim = max_features, output_dim = embedding_dims, 
    input_length = maxlen
    ) %>%
  layer_global_average_pooling_1d() %>%
  layer_dense(1, activation = "sigmoid")

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)


# Fitting -----------------------------------------------------------------

model %>% fit(
  imdb_data$train$x, imdb_data$train$y, 
  batch_size = batch_size,
  epochs = epochs,
  validation_data = list(imdb_data$test$x, imdb_data$test$y)
)
