#' Demonstrates how to write custom layers for Keras.
#'
#' We build a custom activation layer called 'Antirectifier', which modifies the
#' shape of the tensor that passes through it. We need to specify two methods:
#' `compute_output_shape` and `call`.
#'
#' Note that the same result can also be achieved via a Lambda layer.
#' 

#' ## Data Preparation

library(keras)

batch_size <- 128
num_classes <- 10
epochs <- 40

# the data, shuffled and split between train and test sets
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

dim(x_train) <- c(nrow(x_train), 784)
dim(x_test) <- c(nrow(x_test), 784)

x_train <- x_train / 255
x_test <- x_test / 255

cat(nrow(x_train), 'train samples\n')
cat(nrow(x_test), 'test samples\n')

# convert class vectors to binary class matrices
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)

#' ## Antirectifier Layer
#'
#' This is the combination of a sample-wise L2 normalization
#' with the concatenation of the positive part of the input with the negative
#' part of the input. The result is a tensor of samples that are twice as large
#' as the input samples.
#'
#' It can be used in place of a ReLU.
#'
#' Input shape: 2D tensor of shape (samples, n)
#'
#' Output shape: 2D tensor of shape (samples, 2*n)
#'
#' When applying ReLU, assuming that the distribution of the previous output is
#' approximately centered around 0., you are discarding half of your input. This
#' is inefficient.
#'
#' Antirectifier allows to return all-positive outputs like ReLU, without
#' discarding any data.
#'
#' Tests on MNIST show that Antirectifier allows to train networks with twice
#' less parameters yet with comparable classification accuracy as an equivalent
#' ReLU-based network.

# Because our custom layer is written with primitives from the Keras backend
# (`K`), our code can run both on TensorFlow and Theano.
K <- backend()

# Custom layer class
AntirectifierLayer <- R6::R6Class("KerasLayer",
  
  inherit = KerasLayer,
                           
  public = list(
   
    call = function(x, mask = NULL) {
      x <- x - K$mean(x, axis = 1L, keepdims = TRUE)
      x <- K$l2_normalize(x, axis = 1L)
      pos <- K$relu(x)
      neg <- K$relu(-x)
      K$concatenate(c(pos, neg), axis = 1L)
      
    },
     
    compute_output_shape = function(input_shape) {
      input_shape[[2]] <- input_shape[[2]] * 2 
      tuple(input_shape)
    }
  )
)

# create layer wrapper function
layer_antirectifier <- function(object) {
  create_layer(AntirectifierLayer, object)
}


#' ## Define and Train Model

model <- keras_model_sequential()
model %>% 
  layer_dense(units = 256, input_shape = c(784)) %>% 
  layer_antirectifier() %>% 
  layer_dropout(rate = 0.1) %>% 
  layer_dense(units = 256) %>%
  layer_antirectifier() %>% 
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 10, activation = 'softmax')

# compile the model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'rmsprop',
  metrics = c('accuracy')
)

# train the model
model %>% fit(x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  verbose = 1,
  validation_data= list(x_test, y_test)
)

