#' Train a recurrent convolutional network on the IMDB sentiment
#' classification task.
#'  
#' Gets to 0.8498 test accuracy after 2 epochs. 41s/epoch on K520 GPU.

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

# Keras load all data into a list with the following structure:
# List of 2
# $ train:List of 2
# ..$ x:List of 25000
# .. .. [list output truncated]
# .. ..- attr(*, "dim")= int 25000
# ..$ y: num [1:25000(1d)] 1 0 0 1 0 0 1 0 1 0 ...
# $ test :List of 2
# ..$ x:List of 25000
# .. .. [list output truncated]
# .. ..- attr(*, "dim")= int 25000
# ..$ y: num [1:25000(1d)] 1 1 1 1 1 0 0 0 1 1 ...
#
# The x data includes integer sequences, each integer is a word.
# The y data includes a set of integer labels (0 or 1).
# The num_words argument indicates that only the max_fetures most frequent
# words will be integerized. All other will be ignored.
# See help(dataset_imdb)
imdb <- dataset_imdb(num_words = max_features)

# pad the sequences, so they have all the same lenght
# this will conver our dataset into a matrix: each line is a review
# and each column a word on the sequence. 
# we pad the sequences with 0 to the left.
x_train <- imdb$train$x %>%
  pad_sequences(maxlen = maxlen)

x_test <- imdb$test$x %>%
  pad_sequences(maxlen = maxlen)

# Defining the model ------------------------------------------------------

model <- keras_model_sequential()

model %>%
  layer_embedding(max_features, embedding_dims, input_length = maxlen) %>%
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
