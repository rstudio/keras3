#' Use Convolution1D for text classification.
#' 
#' Output after 2 epochs: ~0.89 
#' Time per epoch on CPU (Intel i5 2.4Ghz): 90s
#' Time per epoch on GPU (Tesla K40): 10s
#'

library(keras)

# Set parameters:
max_features <- 5000
maxlen <- 400
batch_size <- 32
embedding_dims <- 50
filters <- 250
kernel_size <- 3
hidden_dims <- 250
epochs <- 2


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

# Pad the sequences, so they have all the same length
# This will convert the dataset into a matrix: each line is a review
# and each column a word on the sequence. 
# Pad the sequences with 0 to the left.
x_train <- imdb$train$x %>%
  pad_sequences(maxlen = maxlen)
x_test <- imdb$test$x %>%
  pad_sequences(maxlen = maxlen)

# Defining Model ------------------------------------------------------

#Initialize model
model <- keras_model_sequential()

model %>% 
  # Start off with an efficient embedding layer which maps
  # the vocab indices into embedding_dims dimensions
  layer_embedding(max_features, embedding_dims, input_length = maxlen) %>%
  layer_dropout(0.2) %>%

  # Add a Convolution1D, which will learn filters
    # Word group filters of size filter_length:
  layer_conv_1d(
    filters, kernel_size, 
    padding = "valid", activation = "relu", strides = 1
  ) %>%
  # Apply max pooling:
  layer_global_max_pooling_1d() %>%

  # Add a vanilla hidden layer:
  layer_dense(hidden_dims) %>%

  # Apply 20% layer dropout
  layer_dropout(0.2) %>%
  layer_activation("relu") %>%

  # Project onto a single unit output layer, and squash it with a sigmoid

  layer_dense(1) %>%
  layer_activation("sigmoid")

# Compile model
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)

# Training ----------------------------------------------------------------

model %>%
  fit(
    x_train, imdb$train$y,
    batch_size = batch_size,
    epochs = epochs,
    validation_data = list(x_test, imdb$test$y)
  )
