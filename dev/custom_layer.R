
library(keras)

K <- backend()


CustomLayer <- R6::R6Class("KerasLayer",
  inherit = KerasLayer,
  
  public = list(
  
    build = function(input_shape) {
      
    },
    
    call = function(x) {
      K$abs(x)
    },
    
    compute_output_shape = function(input_shape) {
      input_shape
    }
  )
)




layer_abs <- function(object, name = NULL, trainable = FALSE) {
  create_layer(CustomLayer, object, list(
    name = name,
    trainable = trainable
  ))
}

model <- keras_model_sequential() 

model %>% 
  layer_dense(units = 32, input_shape = c(10,10)) %>% 
  layer_abs()


