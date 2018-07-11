

#' Base R6 class for Keras models
#'
#' @docType class
#'
#' @format An [R6Class] generator object
#' @section Methods: \describe{
#'   \item{\code{call(inputs,mask)}}{Call the model on an input tensor.}
#'   \item{\code{add_loss(losses, inputs)}}{Add losses to the layer.}
#' }
#'
#' @return [KerasModel].
#' 
#' @note Within the implementation of a custom Keras model you can use `self$` to access
#'   methods and properties of the base Keras model.
#'
#' @export
KerasModel <- R6Class("KerasModel",
             
  public = list(
   
    # Call the model on an input tensor.
    call = function(inputs, mask = NULL) {
      stop("Keras custom models must implement the call function")
    },
 
    # Add losses to the model
    add_loss = function(losses, inputs = NULL) {
      args <- list()
      args$losses <- losses
      args$inputs <- inputs
      do.call(private$wrapper$add_loss, args)
    },

    # back reference to python model that wraps us
    .set_wrapper = function(wrapper) {
      private$wrapper <- wrapper
    },
    
    python_model = function() {
      private$wrapper
    }
  ),
  
  private = list(
    wrapper = NULL
  )
)


#' @export
`$.KerasModel` <- function(x, name) {
  if (name %in% ls(x, all.names = TRUE, sorted = FALSE)) {
    # If name exists in object, return it
    .subset2(x, name)
  } else {
    # otherwise delegate to the python object
    x$python_model()[[name]]
  }
}


#' Create a Keras custom model
#' 
#' @param model_class R6 class of type KerasModel
#' @param ... Additional arguments for R6 constructor 
#' @param name Optional model name
#' 
#' @return A Keras model
#' 
#' @export
keras_model_custom <- function(model_class, ..., name = NULL) {
  
  # verify version
  if (is_tensorflow_implementation() && keras_version() < "2.1.6")
    stop("Custom models require TensorFlow v1.9 or higher")
  else if (!is_tensorflow_implementation() && keras_version() < "2.2.0")
    stop("Custom models require Keras v2.2 or higher")
  
  # create the R model
  args <- list(...)
  r6_model <- do.call(model_class$new, args)
  
  # create the python wrapper (passing the extracted py_wrapper_args)
  python_path <- system.file("python", package = "keras")
  tools <- import_from_path("kerastools", path = python_path)
  model <- do.call(tools$model$RModel, list(
    r_call = r6_model$call,
    name = name
  ))
  
  # set back reference in R model
  r6_model$.set_wrapper(model)
}







