

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


#  Configuraiton of a layer
#  
#  A layer config is a named list containing the configuration of a layer. The
#  same layer can be reinstantiated later (without its trained weights) from
#  this configuration. The config of a layer does not include connectivity
#  information, nor the layer class name (those are handled externally)
#  
#  @inheritParams get_weights
#  
#  @return Named list with layer configuration
#  
#  @seealso [from_config()]
#  
#  @family layer methods
#  
#  @export

# Note: get_config/from_config probably want to be S3 methods for models

get_config <- function(layer) {
  layer$get_config()
}


#  Create a layer from its config
#  
#  This method is the reverse of [get_config()], capable of instantiating the
#  same layer from the config dictionary. It does not handle layer connectivity 
#  (handled by Container), nor weights (handled by [set_weights()]).
#    
#  @param config Named list, typically the output of [get_config()]
#     
#  @seealso [get_config()]
#  
#  @family layer methods
#     
#  @export 
from_config <- function(config, custom_objects = NULL) {
  keras$layers$Layer$from_config(
    config = config,
    custom_objects = custom_objects
  )
}

#  Count the total number of scalars composing the weights.
#  
#  @inheritParams get_weights
#  
#  @return An integer count
#  
#  
#  @family layer methods
#  
#  @export
count_params <- function(layer) {
  layer$count_params()
}







