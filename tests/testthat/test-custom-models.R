context("custom-models")

source("utils.R")

# define model wrapper function
library(keras)

keras_model_simple_mlp <- function(num_classes, 
                                   use_bn = FALSE, use_dp = FALSE, 
                                   name = NULL) {
  
  # define and return a custom model
  keras_model_custom(name = name, function(self) {
    
    # create layers we'll need for the call (this code executes once)
    self$dense1 <- layer_dense(units = 32, activation = "relu")
    self$dense2 <- layer_dense(units = num_classes, activation = "softmax")
    if (use_dp)
      self$dp <- layer_dropout(rate = 0.5)
    if (use_bn)
      self$bn <- layer_batch_normalization(axis = -1)
    
    # implement call (this code executes during training & inference)
    function(inputs, mask = NULL) {
      x <- self$dense1(inputs)
      if (use_dp)
        x <- self$dp(x)
      if (use_bn)
        x <- self$bn(x)
      self$dense2(x)
    }
  })
}

test_succeeds("Use an R-based custom Keras model", {
 
  if (is_tensorflow_implementation() && keras_version() < "2.1.6")
    skip("Custom models require TensorFlow v1.9 or higher")
  else if (!is_tensorflow_implementation() && keras_version() < "2.2.0")
    skip("Custom models require Keras v2.2 or higher")
 
  # create the model 
  model <- keras_model_simple_mlp(10, use_dp = TRUE)
  
  # compile graph
  model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy')
  )
  
  # Generate dummy data
  data <- matrix(runif(1000*100), nrow = 1000, ncol = 100)
  labels <- matrix(round(runif(1000, min = 0, max = 9)), nrow = 1000, ncol = 1)
  
  # Convert labels to categorical one-hot encoding
  one_hot_labels <- to_categorical(labels, num_classes = 10)
  
  # Train the model
  model %>% fit(data, one_hot_labels, epochs=10, batch_size=32)  
  
})