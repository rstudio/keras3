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
Layer <- function(classname, initialize, build = NULL, call = NULL, ...,  
                  inherit = tensorflow::tf$keras$layers$Layer) {
  
  
  defs <- list(
    initialize = initialize,
    build = build,
    call = call
  )
  defs <- Filter(Negate(is.null), defs)
  defs <- append(defs, list(...))
  
  
  # allow using the initialize method
  if ("initialize" %in% names(defs)) {
    if (!is.null(defs$`__init__`))
      stop("You should not specify both __init__ and initialize methods.", call.=FALSE)
    
    defs[["__init__"]] <- defs$initialize
  }
  
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
  
  # makes the function return NULL. `__init__` in python must always return None
  defs$`__init__` <- wrap_return_null(defs$`__init__`)
  
  layer <- reticulate::PyClass(
    classname = classname,
    defs = defs,
    inherit = inherit
  )
  
  f <- function() {
    .args <- as.list(match.call())[-c(1)]
    .args <- .args[names(.args) != "object"]
    create_layer(layer, object, .args)
  }
  formals(f) <- append(
    list(object = quote(expr=)),
    formals(initialize)
  )
  attr(f, "layer") <- layer
  f
}

# makes the function return NULL. `__init__` in python must always return None.
wrap_return_null <- function(fun) {
  e <- new.env(parent = environment(fun))
  e$fun_ <- function() {
    environment(fun) <- environment() # fun must execute in fun_'s exec env.
    do.call(fun, as.list(match.call()[-1]))
    NULL
  }
  formals(e$fun_) <- formals(fun)
  e$fun_
}

