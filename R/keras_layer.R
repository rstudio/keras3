


#' @export
layer_input <- function(shape) {
  keras$layers$Input(tuple(shape))
}

#' @export
layer_dense <- function(x, output_dim, input_dim = NULL, activation = NULL) {
  layer <- keras$layers$Dense(
    output_dim = as.integer(output_dim),
    input_dim = as.integer(input_dim),
    activation = activation
  )
  compose_layer(x, layer)
}

#' @export
layer_activation <- function(x, activation) {
  layer <- keras$layers$Activation(
    activation = activation
  )
  compose_layer(x, layer)
}



# Helper function to compose a layer with an object of type Model or Layer
compose_layer <- function(x, layer) {
  
  # if a sequential is passed then add it to the model
  if (inherits(x, "keras.models.Sequential")) {
    
    x <- clone_model_if_possible(x)
    x$add(layer)
    x
    
  # if a layer is passed then wrap the layer
  } else if (inherits(x, "tensorflow.python.framework.ops.Tensor")) {
    
    call_object(layer, list(x))
    
  # otherwie it's an unexpected type
  } else {
    
    stop("Invalid input to layer function (must be a model or a tensor)",
         call. = FALSE)
  }
}

