


#' Converts a class vector (integers) to binary class matrix.
#' 
#' @details 
#' E.g. for use with [loss_categorical_crossentropy()].
#' 
#' @param y Class vector to be converted into a matrix (integers from 0 to num_classes).
#' @param num_classes Total number of classes.
#' 
#' @return A binary matrix representation of the input.
#' 
#' @export
to_categorical <- function(y, num_classes = NULL) {
  keras$utils$to_categorical(
    y = y,
    num_classes = as_nullable_integer(num_classes)
  )
}

 


as_integer_tuple <- function(x) {
  if (is.null(x))
    x
  else
    tuple(as.list(as.integer(x)))
}

as_nullable_integer <- function(x) {
  if (is.null(x))
    x
  else
    as.integer(x)
}


# Helper function to coerce shape arguments to tuple
normalize_shape <- function(shape) {
  
  # reflect NULL back
  if (is.null(shape))
    return(shape)
  
  # if it's a list or a numeric vector then convert to integer
  if (is.list(shape) || is.numeric(shape)) {
    shape <- lapply(shape, function(value) {
      if (!is.null(value))
        as.integer(value)
      else
        NULL
    })
  }
  
  # coerce to tuple so it's iterable    
  tuple(shape)
}


# Helper function to call a layer
call_layer <- function(layer_function, x, args) {
  
  # remove kwargs that are null
  args$input_shape <- args$input_shape
  args$batch_input_shape = args$batch_input_shape
  args$batch_size <- args$batch_size
  args$dtype <- args$dtype
  args$name <- args$name
  args$trainable <- args$trainable
  args$weights <- args$weights
  
  # call function
  layer <- do.call(layer_function, args)
  
  # compose if we have an x
  if (missing(x) || is.null(x))
    layer
  else
    compose_layer(x, layer)
}



# Helper function to compose a layer with an object of type Model or Layer
compose_layer <- function(x, layer) {
  
  # if a sequential is passed then add it to the model
  if (is_sequential_model(x)) {
    
    x$add(layer)
    x
    
    # if a layer is passed then wrap the layer
  } else if (is_layer(x)) {
    
    layer(x)
    
    # otherwie it's an unexpected type
  } else {
    
    stop("Invalid input to layer function (must be a model or a tensor)",
         call. = FALSE)
  }
}

is_sequential_model <- function(x) {
  inherits(x, "tensorflow.contrib.keras.python.keras.models.Sequential")
}

is_layer <- function(x) {
  inherits(x, "tensorflow.python.framework.ops.Tensor") ||
    inherits(x, "tensorflow.contrib.keras.python.keras.engine.topology.Layer")
}


is_keras_function <- function(f) {
  is.function(f) && identical(environment(f), getNamespace("keras"))
}

resolve_keras_function <- function(f) {
  if (is_keras_function(f))
    f()
  else
    f
}