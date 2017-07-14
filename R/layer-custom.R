

#' Base R6 class for Keras layers
#' 
#' @docType class
#' 
#' @format An [R6Class] generator object
#' #' 
#' @section Methods:
#' \describe{
#'  \item{\code{build(input_shape)}}{Build the layer.}
#'  \item{\code{call(x)}}{Call the layer on an input tensor.}
#'  \item{\code{compute_output_shape(input_shape)}}{Compute the output shape for the layer.}
#' }
#' 
#' @return [KerasLayer].
#' 
#' @export
KerasLayer <- R6Class("KerasLayer",
                         
  public = list(
   
   build = function(input_shape) {
  
   },
   
   call = function(x) {
     stop("Keras custom layers must implement the call function")
   },
   
   compute_output_shape = function(input_shape) {
     input_shape
   }
   
  )
)







