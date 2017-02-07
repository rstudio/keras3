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

# load error message
.load_error_message <- NULL

.onLoad <- function(libname, pkgname) {
  
  # attempt to load keras
  keras  <<- tryCatch(import("keras"), error = function(e) e)
  if (inherits(keras , "error")) {
    .load_error_message <<- keras$message
    keras  <<- NULL
    return()
  }
  
}

.onAttach <- function(libname, pkgname) {
  
  if (is.null(keras)) {
    packageStartupMessage("\n", .load_error_message)
    packageStartupMessage("\nIf you have not yet installed Keras, see https://keras.io\n")
    packageStartupMessage("You should ensure that the version of python where ",
                          "Keras is installed is either the default python ",
                          "on the system PATH or is specified explicitly via the ",
                          "RETICULATE_PYTHON environment variable.\n")
  }
}


