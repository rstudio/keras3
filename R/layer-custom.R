

#' Base R6 class for Keras layers
#'
#' @docType class
#'
#' @format An [R6Class] generator object #'
#' @section Methods: \describe{ \item{\code{build(input_shape)}}{Creates the
#'   layer weights (must be implemented by all layers that have weights)}
#'   \item{\code{call(inputs,mask)}}{Call the layer on an input tensor.}
#'   \item{\code{compute_output_shape(input_shape)}}{Compute the output shape
#'   for the layer.}
#'   \item{\code{add_weight(name,shape,dtype,initializer,regularizer,trainable,constraint)}}{Adds
#'   a weight variable to the layer.} }
#'
#' @return [KerasLayer].
#'
#' @export
KerasLayer <- R6Class("KerasLayer",
             
  public = list(
   
    # Create the layer weights. 
    build = function(input_shape) {
    
    },
   
    # Call the layer on an input tensor.
    call = function(inputs, mask = NULL) {
      stop("Keras custom layers must implement the call function")
    },
   
    # Compute the output shape for the layer.
    compute_output_shape = function(input_shape) {
      input_shape
    },
   
    # Adds a weight variable to the layer.
    add_weight = function(name, shape, dtype = NULL, initializer = NULL,
                          regularizer = NULL, trainable = TRUE, constraint = NULL) {
      
      args <- list()
      args$name <- name
      args$shape <- shape
      args$dtype <- dtype
      args$initializer <- initializer
      args$regularizer <- regularizer
      args$trainable <- trainable
      args$constraint <- constraint
      
      do.call(private$wrapper$add_weight, args)
    },
   
    # back reference to python layer that wraps us
    .set_wrapper = function(wrapper) {
      private$wrapper <- wrapper
    }
  ),
  
  active = list(
    input = function(value) {
      if (missing(value)) return(private$wrapper$input)
      else private$wrapper$input <- value
    },
    output = function(value) {
      if (missing(value)) return(private$wrapper$output)
      else private$wrapper$output <- value
    }
  ),
  
  private = list(
    wrapper = NULL
  )
)









