

#' Install Keras and the TensorFlow backend
#'
#' @inheritParams tensorflow::install_tensorflow
#'
#' @note If you want to do a more customized installation of TensorFlow
#'   (including installing a version that takes advantage of Nvidia GPUs if you
#'   have the correct CUDA libraries installed) you can call the
#'   [install_tensorflow()] function manually before calling `install_keras()`.
#'   See the [article on TensorFlow installation](https://tensorflow.rstudio.com/installation.html) to learn
#'   about more advanced options.
#'
#' @seealso [install_tensorflow()]
#' 
#' @importFrom reticulate py_discover_config
#' @importFrom tensorflow install_tensorflow_extras
#'
#' @export
install_keras <- function(method = c("auto", "virtualenv", "conda", "system"), conda = "auto") {
  
  # see if we already have a version of tensorflow installed into an r-tensorflow environment
  config <- py_discover_config("tensorflow")
  
  # if there is no tensorflow available at all then install it and rediscover the config
  if (is.null(config$required_module_path)) {
    
    install_tensorflow(method = method, conda = conda)
    config <- py_discover_config("tensorflow")
    
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
        
        install_tensorflow(method = method, conda = conda)
        config <- py_discover_config("tensorflow")
        
      }
    }
  }
  
  # at this point we should have an r-tensorflow environment to install keras into
  install_tensorflow_extras("keras", conda = conda)
}