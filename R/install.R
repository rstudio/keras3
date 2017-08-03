

#' Install Keras and the TensorFlow backend
#'
#' @inheritParams tensorflow::install_tensorflow
#'
#' @param method Installation method. By default, "auto" automatically finds a
#'   method that will work in the local environment. Change the default to force
#'   a specific installation method. Note that the "virtualenv" method is not
#'   available on Windows (as this isn't supported by TensorFlow) so "conda"
#'   is the only supported method on windows.
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
#'   If you want to do a fully custom installation of TensorFlow and
#'   Keras using pip (e.g. a shared installation on a server) you can do that
#'   and the keras R package will discover and use that version.
#'   
#'   See the [article on TensorFlow installation](https://tensorflow.rstudio.com/installation.html)
#'   to learn about more advanced installation options.
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
#' # install a specific version of TensorFlow
#' install_keras(tensorflow = c(version = "1.2.1"))
#' 
#' # install with GPU version of TensorFlow 
#' # (NOTE: only do this if you have an Nvidia GPU + CUDA!)
#' install_keras(tensorflow = c(gpu = TRUE))
#' 
#' }
#'
#' @seealso [install_tensorflow()]
#'
#' @importFrom reticulate py_discover_config py_available
#' @importFrom tensorflow install_tensorflow_extras
#'
#' @export
install_keras <- function(method = c("auto", "virtualenv", "conda"), 
                          tensorflow = "default", 
                          conda = "auto") {
  # verify method
  method <- match.arg(method)
  
  # some special handling for windows
  if (is_windows()) {
    
    # conda is the only supported method on windows
    if (identical(method, "auto"))
      method <- "conda"
    
    # avoid DLL in use errors
    if (py_available()) {
      stop("You should call install_keras() only in a fresh ",
           "R session that has not yet initialized Keras and TensorFlow (this is ",
           "to avoid DLL in use errors during installation)")
    }
  }
  
  # build args
  if (identical(tensorflow, "default"))
    args <- list()
  else
    args <- as.list(tensorflow)
  args$method <- method
  args$conda <- conda
  
  # chain keras install onto tf install for virtualenv
  if (identical(method, "virtualenv"))
    args$extra_packages <- "keras"
  
  # perform the installation
  do.call(install_tensorflow, args)
  
  # execute separate keras install for conda
  if (identical(method, "conda"))
    conda_install_keras(conda)
  
  # print success and return
  cat("\nInstallation of Keras complete.\n\n")
  invisible(NULL)
}


# this is a clone of reticulate::conda_install which doesn't pass the 
# --ignore-installed flag (this was causing pip to try to re-install
# scipy from source rather than use the binary version already available
# via conda)
conda_install_keras <- function(conda = "auto") {
  
  # resolve conda binary
  conda <- reticulate::conda_binary(conda)
  
  # packages to install
  envname <- "r-tensorflow"
  packages <- "keras"
  
  # use pip package manager
  condaenv_bin <- function(bin) path.expand(file.path(dirname(conda), bin))
  cmd <- sprintf("%s%s %s && pip install --upgrade %s%s",
                 ifelse(is_windows(), "", ifelse(is_osx(), "source ", "/bin/bash -c \"source ")),
                 shQuote(path.expand(condaenv_bin("activate"))),
                 envname,
                 paste(shQuote(packages), collapse = " "),
                 ifelse(is_windows(), "", ifelse(is_osx(), "", "\"")))
  result <- system(cmd)
    
  
  # check for errors
  if (result != 0L) {
    stop("Error ", result, " occurred installing packages into conda environment ", 
         envname, call. = FALSE)
  }
  
  invisible(NULL)
}



