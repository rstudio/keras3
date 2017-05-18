



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
#'   only, not weights) or the `save_model_hdf5()` function to save the model
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


#' Retrieve tensors for layers with multiple nodes
#' 
#' Whenever you are calling a layer on some input, you are creating a new tensor
#' (the output of the layer), and you are adding a "node" to the layer, linking 
#' the input tensor to the output tensor. When you are calling the same layer 
#' multiple times, that layer owns multiple nodes indexed as 1, 2, 3. These 
#' functions enable you to retreive various tensor properties of layers with 
#' multiple nodes.
#' 
#' @param layer Layer
#' 
#' @param node_index Integer, index of the node from which to retrieve the 
#'   attribute. E.g. `node_index = 1` will correspond to the first time the 
#'   layer was called.
#'   
#' @return A tensor (or list of tensors if the layer has multiple inputs/outputs).
#'   
#' @family layer methods
#'   
#' @export
get_input_at <- function(layer, node_index) {
  layer$get_input_at(as_node_index(node_index))
}

#' @rdname get_input_at
#' @export
get_output_at <- function(layer, node_index) {
  layer$get_output_at(as_node_index(node_index))
}

#' @rdname get_input_at
#' @export
get_input_shape_at <- function(layer, node_index) {
  layer$get_input_shape_at(as_node_index(node_index))
}

#' @rdname get_input_at
#' @export
get_output_shape_at <- function(layer, node_index) {
  layer$get_output_shape_at(as_node_index(node_index))
}


#' @rdname get_input_at
#' @export
get_input_mask_at <- function(layer, node_index) {
  layer$get_input_mask_at(as_node_index(node_index))
}

#' @rdname get_input_at
#' @export
get_output_mask_at <- function(layer, node_index) {
  layer$get_output_mask_at(as_node_index(node_index))
}


#' Reset the states for a layer
#' 
#' @param layer Model or layer
#' 
#' @family layer methods
#' 
#' @export
reset_states <- function(layer) {
  layer$reset_states()
}


as_node_index <- function(node_index) {
  as.integer(node_index-1)
}






