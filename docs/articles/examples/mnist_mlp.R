#' Trains a simple deep NN on the MNIST dataset.
#' 
#' Gets to 98.40% test accuracy after 20 epochs
#' (there is *a lot* of margin for parameter tuning).
#' 2 seconds per epoch on a K520 GPU.
#'

library(keras)

batch_size <- 128
num_classes <- 10
epochs <- 30

# the data, shuffled and split between train and test sets
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

x_train <- array(as.numeric(x_train), dim = c(dim(x_train)[[1]], 784))
x_test <- array(as.numeric(x_test), dim = c(dim(x_test)[[1]], 784))

x_train <- x_train / 255
x_test <- x_test / 255

cat(dim(x_train)[[1]], 'train samples\n')
cat(dim(x_test)[[1]], 'test samples\n')

# convert class vectors to binary class matrices
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)

model <- keras_model_sequential()
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model)

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history <- model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  verbose = 1,
  validation_split = 0.2
)

plot(history)
  
score <- model %>% evaluate(
  x_test, y_test,
  verbose = 0
)
  
cat('Test loss:', score[[1]], '\n')
cat('Test accuracy:', score[[2]], '\n')

