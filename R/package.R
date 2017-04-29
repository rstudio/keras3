
#' @import reticulate
#' @import tensorflow
#' @import methods
#' @import R6
NULL

# Main Keras module
keras <- NULL

.onLoad <- function(libname, pkgname) {
  
  # delay load keras
  keras <<- import("tensorflow.contrib.keras.python.keras", delay_load = function() {
    
  })
  
}

