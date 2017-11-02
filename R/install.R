

#' Install Keras and the TensorFlow backend
#' 
#' Keras and TensorFlow will be installed into an "r-tensorflow" virtual or conda
#' environment. Note that "virtualenv" is not available on Windows (as this isn't
#' supported by TensorFlow).
#'
#' @inheritParams tensorflow::install_tensorflow
#'
#' @param method Installation method ("virtualenv" or "conda")
#' 
#' @param tensorflow TensorFlow version to install. Specify "default" to install
#'   the CPU version of the latest release. Specify "gpu" to install the GPU
#'   version of the latest release.
#'
#'   You can also provide a full major.minor.patch specification (e.g. "1.1.0"),
#'   appending "-gpu" if you want the GPU version (e.g. "1.1.0-gpu").
#'
#'   Alternatively, you can provide the full URL to an installer binary (e.g.
#'   for a nightly binary).
#'
#' @param extra_packages Additional PyPI packages to install along with
#'   Keras and TensorFlow.
#' 
#' @section GPU Installation:
#' 
#' Keras and TensorFlow can be configured to run on either CPUs or GPUs. The CPU 
#' version is much easier to install and configure so is the best starting place 
#' especially when you are first learning how to use Keras. Here's the guidance
#' on CPU vs. GPU versions from the TensorFlow website:
#'
#' - *TensorFlow with CPU support only*. If your system does not have a NVIDIA® GPU, 
#' you must install this version. Note that this version of TensorFlow is typically 
#' much easier to install, so even if you have an NVIDIA GPU, we recommend installing
#' this version first.
#' 
#' - *TensorFlow with GPU support*. TensorFlow programs typically run significantly 
#' faster on a GPU than on a CPU. Therefore, if your system has a NVIDIA® GPU meeting
#' all prerequisites and you need to run performance-critical applications, you should
#' ultimately install this version.
#' 
#' To install the GPU version:
#' 
#' 1) Ensure that you have met all installation prerequisites including installation
#'    of the CUDA and cuDNN libraries as described in [TensorFlow GPU Prerequistes](https://tensorflow.rstudio.com/installation_gpu.html#prerequisites).
#'    
#' 2) Pass `tensorflow = "gpu"` to `install_keras()`. For example:
#' 
#'     ```
#'       install_keras(tensorflow = "gpu")
#'     ````
#' 
#' @section Windows Installation:
#' 
#' The only supported installation method on Windows is "conda". This means that you
#' should install Anaconda 3.x for Windows prior to installing Keras.
#' 
#' @section Custom Installation:
#'   
#' Installing Keras and TensorFlow using `install_keras()` isn't required
#' to use the Keras R package. You can do a custom installation of Keras (and
#' desired backend) as described on the [Keras website](https://keras.io/#installation)
#' and the Keras R package will find and use that version. 
#' 
#' See the documentation on [custom installations](https://tensorflow.rstudio.com/installation.html#custom-installation)
#' for additional information on how version of Keras and TensorFlow are located
#' by the Keras package.
#'
#' @section Additional Packages:
#' 
#' If you wish to add additional PyPI packages to your Keras / TensorFlow environment you 
#' can either specify the packages in the `extra_packages` argument of `install_keras()`, 
#' or alternatively install them into an existing environment using the 
#' [install_tensorflow_extras()] function.
#' 
#' @examples
#' \dontrun{
#'
#' # default installation
#' library(keras)
#' install_keras()
#'
#' # install using a conda environment (default is virtualenv)
#' install_keras(method = "conda")
#'
#' # install with GPU version of TensorFlow
#' # (NOTE: only do this if you have an NVIDIA GPU + CUDA!)
#' install_keras(tensorflow = "gpu")
#'
#' # install a specific version of TensorFlow
#' install_keras(tensorflow = "1.2.1")
#' install_keras(tensorflow = "1.2.1-gpu")
#'
#' }
#'
#' @importFrom reticulate py_available conda_binary
#' 
#' @export
install_keras <- function(method = c("auto", "virtualenv", "conda"), 
                          conda = "auto",
                          tensorflow = "default",
                          extra_packages = NULL) {
  
  # verify method
  method <- match.arg(method)
  
  # some special handling for windows
  if (is_windows()) {
    
    # conda is the only supported method on windows
    method <- "conda"
    
    # confirm we actually have conda
    have_conda <- !is.null(tryCatch(conda_binary(conda), error = function(e) NULL))
    if (!have_conda) {
      stop("Keras installation failed (no conda binary found)\n\n",
           "Install Anaconda for Python 3.x (https://www.continuum.io/downloads#windows)\n",
           "before installing Keras.",
           call. = FALSE)
    }
    
    # avoid DLL in use errors
    if (py_available()) {
      stop("You should call install_keras() only in a fresh ",
           "R session that has not yet initialized Keras and TensorFlow (this is ",
           "to avoid DLL in use errors during installation)")
    }
  }
  
  # extra packages
  extra_packages <- c("keras", extra_packages)
  
  # perform the install
  install_tensorflow(method = method,
                     conda = conda,
                     version = tensorflow,
                     extra_packages = extra_packages)
}



