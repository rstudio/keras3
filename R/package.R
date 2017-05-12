
#' @import reticulate
#' @import tensorflow
#' @import methods
#' @import R6
NULL

# Main Keras module
keras <- NULL

.onLoad <- function(libname, pkgname) {
  
  # delay load keras
  keras <<- import("tensorflow.contrib.keras.python.keras", as = "tensorflow.keras", delay_load = list(
   
    on_load = function() {
      
      # confirm required tf version
      tf_ver <- tf_config()$version
      required_tf_ver <- "1.1"
      if (tf_ver < required_tf_ver) {
        message("Keras loaded from TensorFlow version ", tf_ver, ", however version ",
                required_tf_ver, " is required. Please update TensorFlow.")
      }
    },
    
    on_error = function(e) {
      stop(tf_config()$error_message, call. = FALSE)
    }
     
  ))
}

