context("custom-layers")

source("utils.R")

K <- backend()

# Custom layer class
CustomLayer <- R6::R6Class("KerasLayer",
                                  
  inherit = KerasLayer,
  
  public = list(
    
    output_dim = NULL,
    
    kernel = NULL,
    
    initialize = function(output_dim) {
      self$output_dim <- output_dim
    },
    
    build = function(input_shape) {
      self$kernel <- self$add_weight(
        name = 'kernel', 
        shape = list(input_shape[[1]], self$output_dim),
        initializer = 'uniform',
        trainable = TRUE
      )
    },
    
    call = function(x, mask = NULL) {
      K$dot(x, self$kernel)
    },
    
    compute_output_shape = function(input_shape) {
      list(input_shape[[0]], self$output_dim)
    }
  )
)

# create layer wrapper function
layer_custom <- function(object, output_dim, name = NULL, trainable = TRUE) {
  create_layer(CustomLayer, object, list(
    output_dim = output_dim,
    name = name,
    trainable = trainable
  ))
}

test_succeeds("Use an R-based custom Keras layer", {
  
  model <- keras_model_sequential() 
  model %>%
    layer_dense(32, input_shape = 784, kernel_initializer = initializer_ones()) %>%
    layer_activation('relu') %>%
    layer_custom(output_dim = 10) %>% 
    layer_dense(10) %>%
    layer_activation('softmax')
  
})