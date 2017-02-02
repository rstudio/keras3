

#' Main Keras module
#'
#' Interface to main Kera  module. Provides access to top level classes
#' and functions as well as sub-modules.
#'
#' @format Keras module
#'
#' @export
kr <- NULL


init_modules <- function() {
  
  # attempt to load keras
  kr  <<- tryCatch(import("keras"), error = function(e) e)
  if (inherits(kr , "error")) {
    .load_error_message <<- kr$message
    kr  <<- NULL
    return()
  }
  
}



