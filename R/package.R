
#' @import methods
#' @import R6
#' @importFrom reticulate import dict iterate import_from_path py_call py_capture_output py_get_attr py_has_attr py_is_null_xptr py_to_r r_to_py tuple
#' @importFrom graphics par plot points
#' @importFrom tensorflow tf_version
NULL

# Main Keras module
keras <- NULL

.onLoad <- function(libname, pkgname) {
  
  # determine implementation to use
  implementation <- get_implementation()
  if (identical(implementation, "tensorflow"))
    implementation_module <- "tensorflow.contrib.keras.python.keras"
  else
    implementation_module <- implementation
  
  # delay load keras
  keras <<- import(implementation_module, as = "keras", delay_load = list(
   
    on_load = function() {
      check_implementation_version()
    },
    
    on_error = function(e) {
      stop(tf_config()$error_message, call. = FALSE)
    }
     
  ))
}

get_implementation <- function() {
  getOption("keras.implementation", default = "tensorflow")
}

is_tensorflow_implementation <- function(implementation = get_implementation()) {
  identical(implementation, "tensorflow")
}

is_keras_implementation <- function(implementation = get_implementation()) {
  identical(implementation, "keras")
}

check_implementation_version <- function() {
  
  # get current implementation
  implementation <- get_implementation()
  
  # version variables
  ver <- NULL
  required_ver <- NULL
  
  # define implemetation-specific version/required-version
  if (is_tensorflow_implementation(implementation)) {
    name <- "TensorFlow"
    ver <- tf_version() 
    required_ver <- "1.1"
  } else if (is_keras_implementation(implementation)) {
    name <- "Keras"
    ver <- keras_version()
    required_ver <- "2.0.0"
  }
  
  # check version if we can
  if (!is.null(required_ver)) {
    if (ver < required_ver) {
      message("Keras loaded from ", implementation, " Python module v", ver, ", however version ",
              required_ver, " is required. Please update the ", implementation, " Python package.")
    }
  }
}

keras_version <- function() {
  ver <- keras$`__version__`
  ver <- regmatches(ver, regexec("^([0-9\\.]+).*$", ver))[[1]][[2]]
  package_version(ver)
}



