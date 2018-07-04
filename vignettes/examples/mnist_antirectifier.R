#' Demonstrates how to write custom layers for Keras.
#'
#' We build a custom activation layer called 'Antirectifier', which modifies the
#' shape of the tensor that passes through it. We need to specify two methods:
#' `compute_output_shape` and `call`.
#'
#' Note that the same result can also be achieved via a Lambda layer.
#'

library(keras)

# Data Preparation --------------------------------------------------------

batch_size <- 128
num_classes <- 10
epochs <- 40

# The data, shuffled and split between train and test sets
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# Redimension
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

# Transform RGB values into [0,1] range
x_train <- x_train / 255
x_test <- x_test / 255

cat(nrow(x_train), 'train samples\n')
cat(nrow(x_test), 'test samples\n')

# Convert class vectors to binary class matrices
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)

# Antirectifier Layer -----------------------------------------------------
#'
#' This is the combination of a sample-wise L2 normalization with the
#' concatenation of the positive part of the input with the negative part
#' of the input. The result is a tensor of samples that are twice as large
#' as the input samples.
#'
#' It can be used in place of a ReLU.
#'  Input shape: 2D tensor of shape (samples, n)
#'  Output shape: 2D tensor of shape (samples, 2*n)
#'
#' When applying ReLU, assuming that the distribution of the previous output is
#' approximately centered around 0., you are discarding half of your input. This
#' is inefficient.
#'
#' Antirectifier allows to return all-positive outputs like ReLU, without
#' discarding any data.
#'
#' Tests on MNIST show that Antirectifier allows to train networks with half
#' the parameters yet with comparable classification accuracy as an equivalent
#' ReLU-based network.
#'


# Custom layer class
AntirectifierLayer <- R6::R6Class("KerasLayer",
  
  inherit = KerasLayer,
                           
  public = list(
   
    call = function(x, mask = NULL) {
      x <- x - k_mean(x, axis = 2, keepdims = TRUE)
      x <- k_l2_normalize(x, axis = 2)
      pos <- k_relu(x)
      neg <- k_relu(-x)
      k_concatenate(c(pos, neg), axis = 2)
      
    },
     
    compute_output_shape = function(input_shape) {
      input_shape[[2]] <- input_shape[[2]] * 2L 
      input_shape
    }
  )
)

# Create layer wrapper function
layer_antirectifier <- function(object) {
  create_layer(AntirectifierLayer, object)
}


# Define & Train Model -------------------------------------------------

model <- keras_model_sequential()
model %>% 
  layer_dense(units = 256, input_shape = c(784)) %>% 
  layer_antirectifier() %>% 
  layer_dropout(rate = 0.1) %>% 
  layer_dense(units = 256) %>%
  layer_antirectifier() %>% 
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = num_classes, activation = 'softmax')

# Compile the model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'rmsprop',
  metrics = c('accuracy')
)

# Train the model
model %>% fit(x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  verbose = 1,
  validation_data= list(x_test, y_test)
)
