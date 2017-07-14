
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

lyr <- create_layer(CustomLayer)

layer_abs <- function(object) {
  create_layer(CustomLayer, object)
}

model <- keras_model_sequential() 

model %>% 
  layer_dense(units = 32, input_shape = c(10,10)) %>% 
  layer_abs()


