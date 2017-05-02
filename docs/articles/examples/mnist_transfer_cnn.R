#' Transfer learning toy example:
#' 
#' 1) Train a simple convnet on the MNIST dataset the first 5 digits [0..4].
#' 2) Freeze convolutional layers and fine-tune dense layers
#'    for the classification of digits [5..9].
#'

library(keras)

now <- Sys.time()

batch_size <- 128
num_classes <- 5
epochs <- 5

# input image dimensions
img_rows <- 28
img_cols <- 28

# number of convolutional filters to use
filters <- 32

# size of pooling area for max pooling
pool_size <- 2

# convolution kernel size
kernel_size <- c(3, 3)

# input shape
input_shape <- c(img_rows, img_cols, 1)

# the data, shuffled and split between train and test sets
data <- dataset_mnist()
x_train <- data$train$x
y_train <- data$train$y
x_test <- data$test$x
y_test <- data$test$y

# create two datasets one with digits below 5 and one with 5 and above
x_train_lt5 <- x_train[y_train < 5]
y_train_lt5 <- y_train[y_train < 5]
x_test_lt5 <- x_test[y_test < 5]
y_test_lt5 <- y_test[y_test < 5]

x_train_gte5 <- x_train[y_train >= 5]
y_train_gte5 <- y_train[y_train >= 5] - 5
x_test_gte5 <- x_test[y_test >= 5]
y_test_gte5 <- y_test[y_test >= 5] - 5

# define two groups of layers: feature (convolutions) and classification (dense)
feature_layers <- 
  layer_conv_2d(filters = filters, kernel_size = kernel_size, 
                input_shape = input_shape) %>% 
  layer_activation(activation = 'relu') %>% 
  layer_conv_2d(filters = filters, kernel_size = kernel_size) %>% 
  layer_activation(activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = pool_size) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten()
  


# feature_layers = [
#   Conv2D(filters, kernel_size,
#          padding='valid',
#          input_shape=input_shape),
#   Activation('relu'),
#   Conv2D(filters, kernel_size),
#   Activation('relu'),
#   MaxPooling2D(pool_size=pool_size),
#   Dropout(0.25),
#   Flatten(),
#   ]
# 
# classification_layers = [
#   Dense(128),
#   Activation('relu'),
#   Dropout(0.5),
#   Dense(num_classes),
#   Activation('softmax')
#   ]






