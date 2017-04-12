

# Article named "About Keras Models and Layers# which documents these methods + the multi-input stuff

# or article named:

# "Model and Layer Objects"


# see: https://keras.io/models/about-keras-models/

# see: https://keras.io/layers/about-keras-layers/

# properties (https://github.com/fchollet/keras/blob/master/keras/engine/topology.py#L182-L214)

# name
# input_spec
# trainable
# uses_learning_phase
# input_shape
# output_shape
# inbound_nodes
# input
# output
# input_mask
# output_mask
# trainable_weights
# non_trainable_weights
# weights
# constraints

# methods (https://github.com/fchollet/keras/blob/master/keras/engine/topology.py#L216-L236)

# set_weights(weights)
# get_config()
# from_config(config)
# count_params()
# get_input_at(node_index)
# get_output_at(node_index)
# get_input_shape_at(node_index)
# get_output_shape_at(node_index)
# get_input_mask_at(node_index)
# get_output_mask_at(node_index)


#  Current weights of a layer
#  
#  @param layer Layer 
#  
#  @return Weight values as a list of arrays
#  
#  @family layer methods
#  
#  @export
get_weights <- function(layer) {
  layer$get_weights()
}


#  Set the weights of a layer
#  
#  @inheritParams get_weights
#    
#  @param weights A list of arrays. The number of arrays and their shape must
#    match the number of the dimensions of the weights of the layer (i.e. it
#    should match the output of [get_weights()]).
#    
#  @export
set_weights <- function(layer, weights) {
  layer$set_weights(weights)
}


#' Layer/Model configuration
#' 
#' A layer config is an object returned from `get_config()` that contains the 
#' configuration of a layer or model. The same layer or model can be 
#' reinstantiated later (without its trained weights) from this configuration 
#' using `from_config()`. The config does not include connectivity information, 
#' nor the class name (those are handled externally).
#' 
#' @param layer Layer or model
#' @param config Object with layer or model configuration
#' 
#' @return `get_config()` returns an object with the configuration, 
#'   `from_config()` returns a re-instantation of hte object.
#'   
#' @note Objects returned from `get_config()` are not serializable. Therefore, 
#'   if you want to save and restore a model across sessions, you can use the
#'   `model_to_json()` or `model_to_yaml()` functions (for model configuration
#'   only, not weights) or the `save_model()` function to save the model
#'   configuration and weights to a file.
#'   
#' @family model functions
#' @family layer methods
#'   
#' @export
get_config <- function(layer) {
  
  # call using lower level reticulate functions to prevent converstion to list
  # (the object will remain a python dictionary for full fidelity)
  get_fn <- py_get_attr(layer, "get_config")
  config <- py_call(get_fn)
  
  # set attribute indicating class 
  attr(config, "config_class") <- layer$`__class__`
  config
}


#' @rdname get_config
#' @export
from_config <- function(config) {
  class <- attr(config, "config_class")
  class$from_config(config)
}

#' Layer/Model weights as R arrays
#' 
#' @param layer Layer or model
#' @param weights Weights as R array
#' 
#' @family model persistence
#' @family layer methods
#' 
#' @export
get_weights <- function(layer) {
  layer$get_weights()
}

#' @rdname get_weights
#' @export
set_weights <- function(layer, weights) {
  layer$set_weights(weights)
}


#' Count the total number of scalars composing the weights.
#' 
#' @param layer Layer or model
#'  
#' @return An integer count
#'  
#' @family layer methods
#'  
#' @export
count_params <- function(layer) {
  layer$count_params()
}








