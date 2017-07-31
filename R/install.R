

#' Install Keras and the TensorFlow backend
#' 
#' @inheritParams tensorflow::install_tensorflow
#'
#' @param tensorflow_version TensorFlow version to install (must be either "latest" or a
#'   full major.minor.patch specification, e.g. "1.1.0").
#' @param tensorflow_gpu Install the GPU version of TensorFlow
#' @param tensorflow_package_url URL of the TensorFlow package to install (if not specified
#'   this is determined automatically). Note that if this parameter is provied
#'   then the `tensorflow_version` and `tensorflow_gpu` parameters are ignored.
#'   
#' @importFrom reticulate py_discover_config
#' @importFrom tensorflow install_tensorflow_extras
#'
#' @export
install_keras <- function(method = c("auto", "virtualenv", "conda", "system"), 
                          conda = "auto",
                          tensorflow_version = "latest",
                          tensorflow_gpu = FALSE,
                          tensorflow_package_url = NULL) {
  
  # see if we already have a version of tensorflow installed into an r-tensorflow environment
  config <- py_discover_config("tensorflow")
  
  # helper to install tensorflow and reset the config to the newly installed location
  install_tensorflow_environment <- function() {
    
    install_tensorflow(method = method, 
                       conda = conda, 
                       version = tensorflow_version,
                       gpu = tensorflow_gpu,
                       package_url = tensorflow_package_url)
    
    config <<- py_discover_config("tensorflow")
    
  }
  
  # if there is no tensorflow available at all then install it and rediscover the config
  if (is.null(config$required_module_path)) {
    
    install_tensorflow_environment() 
    
  # otherwise if we don't have a "managed" version of tensorflow then install one
  } else {
    
    # determine which type of tensorflow installation we have
    if (is_windows()) {
      if (config$anaconda)
        type <- "conda"
      else
        type <- "system"
    } else {
      if (nzchar(config$virtualenv_activate))
        type <- "virtualenv"
      else
        type <- "conda"
    }
    
    # if this is a virtualenv or conda based installation then confirm we are within
    # an "r-tensorflow" environment (i.e. installed via install_tensorflow()). if 
    # we aren't then perform an installation of one
    if (type %in% c("virtualenv", "conda")) {
      python_binary <- ifelse(is_windows(), "r-tensorflow\\python.exe", "r-tensorflow/bin/python")
      if (!endsWith(config$python, python_binary)) {
        install_tensorflow_environment() 
      }
    }
  }
  
  # at this point we should have an r-tensorflow environment to install keras into
  install_tensorflow_extras("keras", conda = conda)
}