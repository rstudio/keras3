#' This is an example of using Hierarchical RNN (HRNN) to classify MNIST digits.
#' 
#' HRNNs can learn across multiple levels of temporal hiearchy over a complex sequence.
#' Usually, the first recurrent layer of an HRNN encodes a sentence (e.g. of word vectors)
#' into a  sentence vector. The second recurrent layer then encodes a sequence of
#' such vectors (encoded by the first layer) into a document vector. This
#' document vector is considered to preserve both the word-level and
#' sentence-level structure of the context.
#' 
#' References:
#' - [A Hierarchical Neural Autoencoder for Paragraphs and Documents](https://arxiv.org/abs/1506.01057)
#'   Encodes paragraphs and documents with HRNN.
#'   Results have shown that HRNN outperforms standard RNNs and may play some role in more
#'   sophisticated generation tasks like summarization or question answering.
#' - [Hierarchical recurrent neural network for skeleton based action recognition](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7298714)
#'   Achieved state-of-the-art results on skeleton based action recognition with 3 levels
#'   of bidirectional HRNN combined with fully connected layers.
#' 
#' In the below MNIST example the first LSTM layer first encodes every
#' column of pixels of shape (28, 1) to a column vector of shape (128,). The second LSTM
#' layer encodes then these 28 column vectors of shape (28, 128) to a image vector
#' representing the whole image. A final dense layer is added for prediction.
#' 
#' After 5 epochs: train acc: 0.9858, val acc: 0.9864
#'

library(keras)

# Data Preparation -----------------------------------------------------------------

# Training parameters.
batch_size <- 32
num_classes <- 10
epochs <- 5

# Embedding dimensions.
row_hidden <- 128
col_hidden <- 128

# The data, shuffled and split between train and test sets
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# Reshapes data to 4D for Hierarchical RNN.
x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), 28, 28, 1))
x_train <- x_train / 255
x_test <- x_test / 255

dim_x_train <- dim(x_train)
cat('x_train_shape:', dim_x_train)
cat(nrow(x_train), 'train samples')
cat(nrow(x_test), 'test samples')

# Converts class vectors to binary class matrices
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)

# Define input dimensions
row <- dim_x_train[[2]]
col <- dim_x_train[[3]]
pixel <- dim_x_train[[4]]

# Model input (4D)
input <- layer_input(shape = c(row, col, pixel))

# Encodes a row of pixels using TimeDistributed Wrapper
encoded_rows <- input %>% time_distributed(layer_lstm(units = row_hidden))

# Encodes columns of encoded rows
encoded_columns <- encoded_rows %>% layer_lstm(units = col_hidden)

# Model output
prediction <- encoded_columns %>%
  layer_dense(units = num_classes, activation = 'softmax')

# Define Model ------------------------------------------------------------------------

model <- keras_model(input, prediction)
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'rmsprop',
  metrics = c('accuracy')
)

# Training
model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  verbose = 1,
  validation_data = list(x_test, y_test)
)

# Evaluation
scores <- model %>% evaluate(x_test, y_test, verbose = 0)
cat('Test loss:', scores[[1]], '\n')
cat('Test accuracy:', scores[[2]], '\n')
