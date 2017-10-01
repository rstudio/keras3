#' In this example we will train a DenseNet-40-12 to classify images from the 
#' CIFAR10 small images dataset. This takes ~125s per epoch on a NVIDIA GEFORCE 1080 Ti,
#' so using a GPU is highly recommended.
#' 
#' Final accuracy on test set was 0.9351 versus 0.9300 reported on the 
#' [paper](https://arxiv.org/abs/1608.06993).
#'
#' Beside the `keras` package, you will need to install the `densenet` package.
#' Installation instructions are available [here](https://github.com/dfalbel/densenet).
#'   

# Libraries ---------------------------------------------------------------
library(keras)
library(densenet)

# Parameters --------------------------------------------------------------

batch_size <- 64
epochs <- 300

# Data Preparation --------------------------------------------------------

# see ?dataset_cifar10 for more info
cifar10 <- dataset_cifar10()

# Normalisation
for(i in 1:3){
  mea <- mean(cifar10$train$x[,,,i])
  sds <- sd(cifar10$train$x[,,,i])
  
  cifar10$train$x[,,,i] <- (cifar10$train$x[,,,i] - mea) / sds
  cifar10$test$x[,,,i] <- (cifar10$test$x[,,,i] - mea) / sds
}
x_train <- cifar10$train$x
x_test <- cifar10$test$x

y_train <- to_categorical(cifar10$train$y, num_classes = 10)
y_test <- to_categorical(cifar10$test$y, num_classes = 10)

# Model Definition -------------------------------------------------------

input_img <- layer_input(shape = c(32, 32, 3))
model <- application_densenet(include_top = TRUE, input_tensor = input_img, dropout_rate = 0.2)

opt <- optimizer_sgd(lr = 0.1, momentum = 0.9, nesterov = TRUE)

model %>% compile(
  optimizer = opt,
  loss = "categorical_crossentropy",
  metrics = "accuracy"
)

# Model fitting -----------------------------------------------------------

# callbacks for weights and learning rate
lr_schedule <- function(epoch) {
  
  if(epoch <= 150) {
    return(0.1)
  } else if(epoch > 150 & epoch <= 225){
    return(0.01)
  } else {
    return(0.001)
  }
  
}

lr_reducer <- callback_learning_rate_scheduler(lr_schedule)

history <- model %>% fit(
  x_train, y_train, 
  batch_size = batch_size, 
  epochs = epochs, 
  validation_data = list(x_test, y_test), 
  callbacks = list(
    lr_reducer
  )
)

plot(history)

evaluate(model, x_test, y_test)
