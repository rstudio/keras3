

#' Install Keras and the TensorFlow backend
#'
#' @inheritParams tensorflow::install_tensorflow
#'
#' @param tensorflow Named character vector of additional options to pass to
#'   [install_tensorflow()]. If this argument is "default" then a previous 
#'   installation of TensorFlow will be used if available. Otherwise, a new
#'   installation will be performed using the specified options.
#'
#' @note If you want to do a more customized installation of TensorFlow
#'   (including installing a version that takes advantage of Nvidia GPUs if you
#'   have the correct CUDA libraries installed) you can pass additional options
#'   to the [install_tensorflow()] function using the `tensorflow` argument.
#'
#'   You can also call [install_tensorflow()] prior to calling `install_keras()`
#'   and Keras will be installed alongside the version of TensorFlow installed
#'   in this fashion. 
#'   
#'   Finally, if you want to do a fully custom installation of TensorFlow and
#'   Keras using pip (e.g. a shared installation on a server) you can do that
#'   and the keras R package will discover and use that version.
#'   
#'   See the [article on TensorFlow installation]
#'   (https://tensorflow.rstudio.com/installation.html) to learn
#'   about more advanced installation options.
#'   
#' @examples 
#' \dontrun{
#' # default installation
#' library(keras)
#' install_keras()
#' 
#' # install using a conda environment (default is virtualenv)
#' install_keras(method = "conda")
#' 
#' # install with GPU version of TensorFlow 
#' # (NOTE: only do this if you have an Nvidia GPU + CUDA!)
#' install_keras(tensorflow = c(gpu = TRUE))
#' 
#' # install TensorFlow w/ options first then install Keras
#' install_tensorflow(version = "1.2.1")
#' install_keras()
#' }   
#'
#' @seealso [install_tensorflow()]
#'
#' @importFrom reticulate py_discover_config py_available
#' @importFrom tensorflow install_tensorflow_extras
#'
#' @export
install_keras <- function(method = c("auto", "virtualenv", "conda", "system"), 
                          tensorflow = "default", conda = "auto") {
  
  # ensure we call this in a fresh session on windows (avoid DLL
  # in use errors)
  if (is_windows() && py_available()) {
    stop("You should call install_keras() only in a fresh ",
         "R session that has not yet initialized Keras and TensorFlow (this is ",
         "to avoid DLL in use errors during installation)")
  }
  
  # utility function to install a "managed" version of tensorflow
  install_managed_tensorflow <- function() {
    if (identical(tensorflow, "default"))
      args <- list()
    else
      args <- as.list(tensorflow)
    args$method <- method
    args$conda <- conda
    do.call(install_tensorflow, args)
  }
  
  # if tensorflow options were provied then install tensorflow and use the specifed version
  if (!identical(tensorflow, "default")) {
    install_managed_tensorflow()
    if (identical(method, "virtualenv"))
      use_virtualenv("r-tensorflow")
    else if (identical(method, "conda"))
      use_condaenv("r-tensorflow")
  }
  
  # see if we already have a version of tensorflow installed into an r-tensorflow environment
  config <- py_discover_config("tensorflow")
  
  # if there is no tensorflow available at all then install it and rediscover the config
  if (is.null(config$required_module_path)) {
    
    install_managed_tensorflow()
    
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
    
    # if this is a virtualenv or conda based installation then do some extra checking
    if (type %in% c("virtualenv", "conda")) {
      
      # confirm we are in an "r-tensorflow" environment (i.e. installed via install_tensorflow()). if 
      # we aren't then perform an installation of one
      python_binary <- ifelse(is_windows(), "r-tensorflow\\python.exe", "r-tensorflow/bin/python")
      if (!grepl(paste0(python_binary, "$"), config$python)) {
        
        install_managed_tensorflow()
       
      # confirm that what we found matches any explicit method (if not then install using
      # the requested method) 
      } else if (!identical(method, "auto") && (!identical(method, type))) {
        
        install_managed_tensorflow()
        
      }
    }
  }
  
  # at this point we should have an r-tensorflow environment to install keras into
  install_tensorflow_extras("keras", conda = conda)
  
  cat("\nInstallation of Keras complete.\n\n")
  invisible(NULL)
}