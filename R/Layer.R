#' Create a custom Layer
#' 
#' @param classname the name of the custom Layer.
#' @param defs definition of methods.
#' @param inherit innherit from the base Keras Layer. 
#' 
#' @examples
#' \dontrun{
#' 
#' layer_dense2 <- Layer(
#'   "Dense2",
#'   list(
#'     
#'     initialize = function(units) {
#'       self$units <- units
#'     },
#'     
#'     build = function(input_shape) {
#'       self$kernel <- self$add_weight(
#'         name = "kernel",
#'         shape = list(input_shape[[2]], self$units),
#'         initializer = "uniform",
#'         trainable = TRUE
#'       )
#'     },
#'     
#'     call = function(x) {
#'       tensorflow::tf$matmul(x, self$kernel)
#'     },
#'     
#'     compute_output_shape = function(input_shape) {
#'       list(input_shape[[1]], self$units)
#'     }
#'     
#'   )
#' )
#' 
#' 
#' }
#' 
#' 
#' @export
Layer <- function(classname, defs, inherit = tensorflow::tf$keras$layers$Layer) {
  
  # allow using the initialize method
  if ("initialize" %in% names(defs)) {
    if (!is.null(defs$`__init__`))
      stop("You should specidy both __init__ and initialize methods.", call.=FALSE)
    
    defs[["__init__"]] <- defs$initialize
  }
  
  # make `__init__` always return NULL
  # init <- defs[["__init__"]]
  # defs[["__init__"]] <- function(...) {
  #   init(...)
  #   NULL
  # }
  
  # automatically add the `self` argument
  defs <- lapply(defs, function(x) {
    
    if (inherits(x, "function")) {
     formals(x) <- append(
       pairlist(self = NULL),
       formals(x)
     )
    }
    
    x
  })
  
  
  layer <- reticulate::PyClass(
    classname = classname,
    defs = defs,
    inherit = inherit
  )
  
  function(object, ...) {
    create_layer(layer, object, list(...))
  }
}
