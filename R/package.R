
#' @import reticulate
#' @import tensorflow
#' @import methods
#' @import R6
NULL

# Main Keras module
keras <- NULL

.onLoad <- function(libname, pkgname) {
  
  # delay load keras
  keras <<- import("tensorflow.contrib.keras.python.keras", as = "tensorflow.keras", delay_load = function() {
  
    # confirm required tf version
    tf_version <- tensorflow_version()
    required_tf_version <- "1.1"
    if (tf_version < required_tf_version) {
      message("Keras loaded from TensorFlow version ", tf_version, ", however version ",
              required_tf_version, " is required. Please update TensorFlow.")
    }
                       
  })
}

