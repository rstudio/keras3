#' This is a reproduction of the IRNN experiment with pixel-by-pixel sequential 
#' MNIST in "A Simple Way to Initialize Recurrent Networks of Rectified Linear Units"
#' by Quoc V. Le, Navdeep Jaitly, Geoffrey E. Hinton
#' 
#' arxiv:1504.00941v2 [cs.NE] 7 Apr 2015
#' http://arxiv.org/pdf/1504.00941v2.pdf
#' 
#' Optimizer is replaced with RMSprop which yields more stable and steady
#' improvement.
#' 
#' Reaches 0.93 train/test accuracy after 900 epochs
#' This corresponds to roughly 1687500 steps in the original paper.

library(keras)

# Data Preparation ---------------------------------------------------------------

batch_size <- 32
num_classes <- 10
epochs <- 200
hidden_units <- 100

img_rows <- 28
img_cols <- 28

learning_rate <- 1e-6
clip_norm <- 1.0

# The data, shuffled and split between train and test sets
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

x_train <- array_reshape(x_train, c(nrow(x_train), img_rows * img_cols, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), img_rows * img_cols, 1))
input_shape <- c(img_rows, img_cols, 1)

# Transform RGB values into [0,1] range
x_train <- x_train / 255
x_test <- x_test / 255

cat('x_train_shape:', dim(x_train), '\n')
cat(nrow(x_train), 'train samples\n')
cat(nrow(x_test), 'test samples\n')

# Convert class vectors to binary class matrices
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)

# Define Model ------------------------------------------------------------------

model <- keras_model_sequential()
model %>% 
  layer_simple_rnn(units = hidden_units,
                   kernel_initializer = initializer_random_normal(stddev = 0.01),
                   recurrent_initializer = initializer_identity(gain = 1.0),
                   activation = 'relu',
                   input_shape = dim(x_train)[-1]) %>% 
  layer_dense(units = num_classes) %>% 
  layer_activation(activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(lr = learning_rate),
  metrics = c('accuracy')
)
 
# Training & Evaluation ---------------------------------------------------------

cat("Evaluate IRNN...\n")
model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  verbose = 1,
  validation_data = list(x_test, y_test)
)
  
scores <- model %>% evaluate(x_test, y_test, verbose = 0)
cat('IRNN test score:', scores[[1]], '\n')
cat('IRNN test accuracy:', scores[[2]], '\n')
