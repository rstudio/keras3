#' keras for R
#' 
#' \href{https://keras.io}{Keras}Keras is a high-level neural networks library
#' built on top of TensorFlow. It was developed with a focus on enabling fast
#' experimentation. 
#' 
#' @docType package
#' @name keras
#' @import reticulate
NULL

# Main Keras module
keras <- NULL

.onLoad <- function(libname, pkgname) {
  
  # delay load keras
  keras <<- import("keras", delay_load = TRUE)
  
}

