#' Base R6 class for Keras wrappers
#'
#' @docType class
#'
#' @format An [R6Class] generator object
#' 
#' @section Methods: \describe{ 
#'   \item{\code{build(input_shape)}}{Builds the wrapped layer. 
#'   Subclasses can extend this to perform custom operations on that layer.}
#'   \item{\code{call(inputs,mask)}}{Calls the wrapped layer on an input tensor.}
#'   \item{\code{compute_output_shape(input_shape)}}{Computes the output shape
#'   for the wrapped layer.}
#'   \item{\code{add_loss(losses, inputs)}}{Subclasses can use this to add losses to the wrapped layer.}
#'   \item{\code{add_weight(name,shape,dtype,initializer,regularizer,trainable,constraint)}}{Subclasses can use this to add weights to the wrapped layer.} }
#'
#' @return [KerasWrapper].
#'
#' @export
KerasWrapper <- R6::R6Class(
  "KerasWrapper",
  
  public = list(
    build = function(input_shape) {
      if (!private$py_wrapper$layer$built) private$py_wrapper$layer$build(input_shape)
    },
    
    call = function(inputs, mask = NULL) {
      private$py_wrapper$layer$call(inputs)
    },
    
    compute_output_shape = function(input_shape) {
      private$py_wrapper$layer$compute_output_shape(input_shape)
    },
    
    add_loss = function(losses, inputs = NULL) {
      args <- list()
      args$losses <- losses
      args$inputs <- inputs
      do.call(private$py_wrapper$layer$add_loss, args)
    },
    
    add_weight = function(name,
                          shape,
                          dtype = NULL,
                          initializer = NULL,
                          regularizer = NULL,
                          trainable = TRUE,
                          constraint = NULL) {
      args <- list()
      args$name <- name
      args$shape <- shape
      args$dtype <- dtype
      args$initializer <- initializer
      args$regularizer <- regularizer
      args$trainable <- trainable
      args$constraint <- constraint

      do.call(private$py_wrapper$layer$add_weight, args)
    },
    
    .set_py_wrapper = function(py_wrapper) {
      private$py_wrapper <- py_wrapper
    },
    
    python_wrapper = function() {
      private$py_wrapper
    }
  ),
  
  active = list(
    input = function(value) {
      if (missing(value))
        return(private$py_wrapper$input)
      else
        private$py_wrapper$input <- value
    },
    output = function(value) {
      if (missing(value))
        return(private$py_wrapper$output)
      else
        private$py_wrapper$output <- value
    }
  ),
  
  private = list(py_wrapper = NULL)
)

#' Create a Keras Wrapper
#' 
#' @param wrapper_class R6 class of type KerasWrapper
#' @param object Object to compose layer with. This is either a 
#' [keras_model_sequential()] to add the layer to, or another Layer which
#' this layer will call.
#' @param args List of arguments to layer constructor function 
#' 
#' @return A Keras wrapper
#' 
#' @note The `object` parameter can be missing, in which case the 
#' layer is created without a connection to an existing graph.
#' 
#' @export
create_wrapper <- function(wrapper_class, object, args = list()) {
  
  args$layer <- args$layer
  args$input_shape <- args$input_shape
  args$batch_input_shape <- args$batch_input_shape
  args$batch_size <- args$batch_size
  args$dtype <- args$dtype
  args$name <- args$name
  args$trainable <- args$trainable
  args$weights <- args$weights
  
  common_arg_names <- c("layer", "input_shape", "batch_input_shape", "batch_size",
                          "dtype", "name", "trainable", "weights")
  py_wrapper_args <- args[common_arg_names]
  py_wrapper_args[sapply(py_wrapper_args, is.null)] <- NULL
  for (arg in names(py_wrapper_args))
  args[[arg]] <- NULL
    
  r6_wrapper <- do.call(wrapper_class$new, args)
    
  python_path <- system.file("python", package = "keras")
  tools <- reticulate::import_from_path("kerastools", path = python_path)
  py_wrapper_args$r_build <- r6_wrapper$build
  py_wrapper_args$r_call <-  r6_wrapper$call
  py_wrapper_args$r_compute_output_shape <- r6_wrapper$compute_output_shape
  py_wrapper <- do.call(tools$wrapper$RWrapper, py_wrapper_args)
    
  r6_wrapper$.set_py_wrapper(py_wrapper)  


  if (missing(object) || is.null(object))
    r6_wrapper
  else
    invisible(compose_layer(object, py_wrapper))
}

