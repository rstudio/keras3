
#' @import reticulate
#' @import tensorflow
#' @import R6
NULL

# Main Keras module
keras <- NULL

.onLoad <- function(libname, pkgname) {
  
  # delay load keras
  keras <<- import("tensorflow.contrib.keras.python.keras", delay_load = function() {
    
  })
  
}

# ensure that keras loads (this allows us to force binding to a version of python
# that has keras available)
ensure_keras <- function() {
  keras:::keras$`__version__`
}
