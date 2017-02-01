#' keras for R
#' 
#' \href{https://keras.io}{Keras}Keras is a high-level neural networks library
#' built on top of TensorFlow. It was developed with a focus on enabling fast
#' experimentation. 
#' 
#' @docType package
#' @name keras
#' @import tensorflow
NULL


# load error message
.load_error_message <- NULL

.onLoad <- function(libname, pkgname) {
  init_modules()
}

.onAttach <- function(libname, pkgname) {
  
  if (is.null(kr)) {
    packageStartupMessage("\n", .load_error_message)
    packageStartupMessage("\nIf you have not yet installed Keras, see https://keras.io\n")
    packageStartupMessage("You should ensure that the version of python where ",
                          "Keras is installed is either the default python ",
                          "on the system PATH or is specified explicitly via the ",
                          "TENSORFLOW_PYTHON environment variable.\n")
  }
}