#' Train a recurrent convolutional network on the IMDB sentiment
#' classification task.
#'  
#' Achieves 0.8498 test accuracy after 2 epochs. 41s/epoch on K520 GPU.

library(keras)

# Parameters --------------------------------------------------------------

# Embedding
max_features = 20000
maxlen = 100
embedding_size = 128

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 30
epochs = 2

# Data Preparation --------------------------------------------------------

# The x data includes integer sequences, each integer is a word
# The y data includes a set of integer labels (0 or 1)
# The num_words argument indicates that only the max_fetures most frequent
# words will be integerized. All other will be ignored.
# See help(dataset_imdb)
imdb <- dataset_imdb(num_words = max_features)
# Keras load all data into a list with the following structure:
str(imdb)

# Pad the sequences to the same length
  # This will convert our dataset into a matrix: each line is a review
  # and each column a word on the sequence
# We pad the sequences with 0s to the left
x_train <- imdb$train$x %>%
  pad_sequences(maxlen = maxlen)
x_test <- imdb$test$x %>%
  pad_sequences(maxlen = maxlen)

# Defining Model ------------------------------------------------------

model <- keras_model_sequential()

model %>%
  layer_embedding(max_features, embedding_size, input_length = maxlen) %>%
  layer_dropout(0.25) %>%
  layer_conv_1d(
    filters, 
    kernel_size, 
    padding = "valid",
    activation = "relu",
    strides = 1
  ) %>%
  layer_max_pooling_1d(pool_size) %>%
  layer_lstm(lstm_output_size) %>%
  layer_dense(1) %>%
  layer_activation("sigmoid")

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)

# Training ----------------------------------------------------------------

model %>% fit(
  x_train, imdb$train$y,
  batch_size = batch_size,
  epochs = epochs,
  validation_data = list(x_test, imdb$test$y)
)
