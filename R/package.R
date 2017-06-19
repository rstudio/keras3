
#' @import methods
#' @import R6
#' @importFrom reticulate import dict iterate import_from_path py_call py_capture_output py_get_attr py_has_attr py_is_null_xptr py_to_r r_to_py tuple
#' @importFrom graphics par plot points
#' @importFrom tensorflow tf_version
NULL

# Main Keras module
keras <- NULL

.onLoad <- function(libname, pkgname) {
  
  # delay load keras
  keras <<- import("tensorflow.contrib.keras.python.keras", as = "tensorflow.keras", delay_load = list(
   
    on_load = function() {
      check_tf_version()
    },
    
    on_error = function(e) {
      stop(tf_config()$error_message, call. = FALSE)
    }
     
  ))
}

check_tf_version <- function() {
  tf_ver <- tf_config()$version
  required_tf_ver <- "1.1"
  if (tf_ver < required_tf_ver) {
    message("Keras loaded from TensorFlow version ", tf_ver, ", however version ",
            required_tf_ver, " is required. Please update TensorFlow.")
  }
}