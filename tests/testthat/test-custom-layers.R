context("custom-layers")

source("utils.R")

# Custom layer class
CustomLayer <- R6::R6Class("CustomLayer",
                                  
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
        shape = list(input_shape[[2]], self$output_dim),
        initializer = initializer_random_normal(),
        trainable = TRUE
      )
    },
    
    call = function(x, mask = NULL) {
      self$add_loss(list(k_constant(5)))
      k_dot(x, self$kernel)
    },
    
    compute_output_shape = function(input_shape) {
      list(input_shape[[1]], self$output_dim)
    }
  )
)

# create layer wrapper function
layer_custom <- function(object, output_dim, name = NULL, trainable = TRUE) {
  create_layer(CustomLayer, object, list(
    output_dim = as.integer(output_dim),
    name = name,
    trainable = trainable
  ))
}


test_succeeds("Use an R-based custom Keras layer", {
  skip_if_tensorflow_implementation()
  model <- keras_model_sequential()
  model %>% 
    layer_dense(units = 32, input_shape = c(32,32)) %>% 
    layer_custom(output_dim = 32)
})