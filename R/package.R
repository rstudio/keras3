
#' @import methods
#' @import R6
#' @importFrom reticulate import dict iterate import_from_path py_call py_capture_output py_get_attr py_has_attr py_is_null_xptr py_to_r r_to_py tuple
NULL

# Main Keras module
keras <- NULL

# startup messages
.startupMessage <- NULL

.onLoad <- function(libname, pkgname) {
  
  # delay load keras
  keras <<- import("tensorflow.contrib.keras.python.keras", as = "tensorflow.keras", delay_load = list(
   
    on_load = function() {
      
      # confirm required tf version
      tf_ver <- tf_config()$version
      required_tf_ver <- "1.1"
      if (tf_ver < required_tf_ver) {
        .startupMessage <- paste0(
          "Keras loaded from TensorFlow version ", tf_ver, ", however version ",
          required_tf_ver, " is required. Please update TensorFlow.")
      }
    },
    
    on_error = function(e) {
      stop(tf_config()$error_message, call. = FALSE)
    }
     
  ))
}

.onAttach <- function(libname, pkgname) {
  if (!is.null(.startupMessage))
    packageStartupMessage(.startupMessage)
}

