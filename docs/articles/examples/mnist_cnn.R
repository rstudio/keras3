
#' Trains a simple convnet on the MNIST dataset.
#' 
#' Gets to 99.25% test accuracy after 12 epochs
#' (there is still a lot of margin for parameter tuning).
#' 16 seconds per epoch on a GRID K520 GPU.

library(keras)

batch_size <- 128
num_classes <- 10
epochs <- 1

# input image dimensions
img_rows <- 28
img_cols <- 28

# the data, shuffled and split between train and test sets
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

x_train <- array(as.numeric(x_train), dim = c(dim(x_train)[[1]], img_rows, img_cols, 1))
x_test <- array(as.numeric(x_test), dim = c(dim(x_test)[[1]], img_rows, img_cols, 1))
input_shape <- c(img_rows, img_cols, 1)

x_train <- x_train / 255
x_test <- x_test / 255

cat('x_train_shape:', dim(x_train))
cat(dim(x_train)[[1]], 'train samples')
cat(dim(x_test)[[1]], 'test samples')

# convert class vectors to binary class matrices
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)

# define model
model <- keras_model_sequential()
model %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                input_shape = input_shape) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = num_classes, activation = 'softmax')

# compile model
model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

# train and evaluate
model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  verbose = 1,
  validation_data = list(x_test, y_test)
)
score <- model %>% evaluate(
  x_test, y_test, verbose = 0
)

cat('Test loss:', score[[1]])
cat('Test accuracy:', score[[2]])


