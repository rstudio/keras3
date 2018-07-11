context("custom-models")

source("utils.R")

# Custom model class
SimpleMLP <- R6::R6Class("SimpleMLP",
                         
  inherit = KerasModel,
  
  public = list(
    
    num_classes = NULL,
    use_bn = NULL,
    use_dp = NULL,
    dense1 = NULL,
    dense2 = NULL,
    dp = NULL,
    bn = NULL,
    
    initialize = function(num_classes, use_bn = FALSE, use_dp = FALSE) {
      self$num_classes <- num_classes
      self$use_bn <- use_bn
      self$use_dp <- use_dp
      self$dense1 <- layer_dense(units = 32, activation = "relu")
      self$dense2 <- layer_dense(units = num_classes, activation = "softmax")
      if (self$use_dp)
        self$dp <- layer_dropout(rate = 0.5)
      if (self$use_bn)
        self$bn <- layer_batch_normalization(axis = -1)
    },
    
    call = function(inputs, mask = NULL) {
      x <- self$dense1(inputs)
      if (self$use_dp)
        x <- self$dp(x)
      if (self$use_bn)
        x <- self$bn(x)
      self$dense2(x)
    }
  )
)

# define model wrapper function
keras_model_simple_mlp <- function(num_classes, use_bn = FALSE, use_dp = FALSE, name = NULL) {
  keras_model_custom(SimpleMLP, 
    num_classes = num_classes,
    use_bn = use_bn,
    use_dp = use_dp,
    name = name
  )
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