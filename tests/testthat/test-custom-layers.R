context("custom-layers")

source("utils.R")

K <- backend()

# Custom layer class
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

test_succeeds("Use an R-based custom Keras layer", {
  model <- keras_model_sequential()
  model %>% 
    layer_dense(units = 256, input_shape = c(784)) %>% 
    layer_antirectifier() %>% 
    layer_dropout(rate = 0.1) %>% 
    layer_dense(units = 256) %>%
    layer_antirectifier() %>% 
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = 10, activation = 'softmax')
})