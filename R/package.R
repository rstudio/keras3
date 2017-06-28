
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
  implementation <- get_keras_implementation(default = NULL)
  
  # determine backend to use
  backend <- get_keras_backend()
  
  # set KERAS_BACKEND environment variable 
  if (!is.null(backend))
    Sys.setenv(KERAS_BACKEND = backend)
  
  # fully resolve implementation if it's not yet provided
  if (is.null(implementation)) {
    # if there is a backend then this implies 'keras' implementation
    if (!is.null(backend)) {
      implementation <- "keras"
      Sys.setenv(KERAS_IMPLEMENTATION = "keras")
    # otherwise implementation is 'tensorflow'
    } else {
      implementation <- "tensorflow"
    }
  }
  
  # set the implementation module
  if (identical(implementation, "tensorflow"))
    implementation_module <- "tensorflow.contrib.keras.python.keras"
  else
    implementation_module <- implementation
  
  # if KERAS_PYTHON is defined then forward it to RETICULATE_PYTHON
  keras_python <- get_keras_python()
  if (!is.null(keras_python))
    Sys.setenv(RETICULATE_PYTHON = keras_python)
  
  # delay load keras
  keras <<- import(implementation_module, as = "keras", delay_load = list(
  
    priority = 10,
     
    on_load = function() {
      check_implementation_version()
    },
    
    on_error = function(e) {
      stop(tf_config()$error_message, call. = FALSE)
    }
     
  ))
  
  # until we depend on a version of reticulate that implements the 'priority'
  # field we manually overwrite the 'delay_load_module` so that we can 
  # still find the right python when not using the tensorflow back-end
  globals <- get(".globals", envir = getNamespace("reticulate"))
  globals$delay_load_module <- implementation_module
}

get_keras_implementation <- function(default = "tensorflow") {
  get_keras_option("keras.implementation", default = default)
}

get_keras_backend <- function(default = NULL) {
  get_keras_option("keras.backend", default = default)
}

get_keras_python <- function(default = NULL) {
  get_keras_option("keras.python", default = default)
}

get_keras_option <- function(name, default = NULL) {
  
  # first check the option
  value <- getOption(name, default = NA)
  if (!is.na(value)) {
    value
  } else {
    env_var_name <- gsub(".", "_", toupper(name), fixed = TRUE)
    value <- Sys.getenv(env_var_name, unset = NA)
    if (!is.na(value))
      value
    else
      default
  }
}


is_tensorflow_implementation <- function(implementation = get_keras_implementation()) {
  grepl("^tensorflow", implementation)
}

is_keras_implementation <- function(implementation = get_keras_implementation()) {
  identical(implementation, "keras")
}

check_implementation_version <- function() {
  
  # get current implementation
  implementation <- get_keras_implementation()
  
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



