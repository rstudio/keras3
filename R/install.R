#' Install Keras
#'
#' This function will install Keras along with a selected backend, including all Python dependencies.
#'
#' @param envname Name of or path to a Python virtual environment
#' @param extra_packages Additional Python packages to install alongside Keras
#' @param python_version Passed on to `reticulate::virtualenv_starter()`
#' @param backend Which backend(s) to install. Accepted values include `"tensorflow"`, `"jax"` and `"torch"`
#' @param gpu whether to install a GPU capable version of the backend.
#' @param restart_session Whether to restart the R session after installing (note this will only occur within RStudio).
#' @param ... reserved for future compatibility.
#'
#' @returns No return value, called for side effects.
#'
#' @seealso [`tensorflow::install_tensorflow()`]
#' @export
install_keras <- function(
    envname = "r-keras", ...,
    extra_packages = c("scipy", "pandas", "Pillow", "pydot", "ipython", "tensorflow_datasets"),
    python_version = ">=3.9,<=3.11",
    # backend = "tensorflow",
    backend = c("tensorflow", "jax", "torch"),
    # backend = "tf-nightly",
    gpu = NA,
    restart_session = TRUE) {

  if (is.na(gpu)) {
    has_nvidia_gpu <- function()
      tryCatch(as.logical(length(system("lspci | grep -i nvidia", intern = TRUE))),
               warning = function(w) FALSE)
    # don't install tensorflow-metal until it's been updated
    # https://pypi.org/project/tensorflow-metal/#history
    gpu <- (is_linux() && has_nvidia_gpu()) ## ||  is_mac_arm64()
  }

  # keras requires tensorflow be installed still.
  if(!any(grepl("tensorflow|tf-nightly", backend)))
    backend <- c("tensorflow", backend)

  if (isTRUE(gpu)) {
    message("Installing GPU components")
    if (is_mac_arm64()) {
      jax <- c("jax-metal") # ??? do we still need this, "ml-dtypes==0.2.*")
      ## installation of 'tensorflow-metal' is disabled until a new version that
      ## is compatible with TF v2.16 is released.
      # tensorflow <- c("tensorflow", "tensorflow-metal")
    } else if (is_linux()) {
      jax <- c("jax[cuda12_pip]", "-f",
         "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
      tensorflow <- "tensorflow[and-cuda]"
    }
  } else { # no GPU
    jax <- "jax[cpu]"
    tensorflow <- if(is_linux()) "tensorflow-cpu" else "tensorflow"
  }

  # The "numpy" backend requires that "jax" be installed
  # if("jax" %in% backend && !is.null(extra_packages))
  #   # undeclared dependency, import fails otherwise
  #   append(extra_packages) <- "packaging"

  backend <- unlist(lapply(backend, function(name)
    switch(name,
           jax = jax,
           tensorflow = tensorflow,
           "tf-nightly" = local({
             tensorflow <- sub("tensorflow", "tf-nightly", x = tensorflow, fixed = TRUE)
             replace_val(tensorflow, "tf-nightly-metal", "tensorflow-metal")
           }),
           name)
    ))

  reticulate::virtualenv_create(
    envname = envname,
    version = python_version,
    force = identical(envname, "r-keras"),
    packages = NULL
  )
  extra_packages <- unique(extra_packages)
  if (length(extra_packages))
    reticulate::py_install(extra_packages, envname = envname)

  if (length(backend))
    reticulate::py_install(backend, envname = envname)

  if(gpu && is_linux()) {
    configure_cudnn_symlinks(envname = envname)
  }

  reticulate::py_install("keras==3.*", envname = envname)
                         #, pip_ignore_installed = TRUE)

  message("Finished installing Keras!")
  if (restart_session && requireNamespace("rstudioapi", quietly = TRUE) &&
    rstudioapi::hasFun("restartSession")) {
    rstudioapi::restartSession()
  }

  invisible(NULL)
}

is_linux <- function() {
  unname(Sys.info()[["sysname"]] == "Linux")
}

#' Configure a Keras backend
#'
#' @param backend string, can be `"tensorflow"`, `"jax"`, `"numpy"`, or `"torch"`.
#'
#' @details
#' These functions allow configuring which backend keras will use.
#' Note that only one backend can be configured at a time.
#'
#' The function should be called after `library(keras3)` and before calling
#' other functions within the package (see below for an example).
#'
#' There is experimental support for changing the backend after keras has initialized.
#' using `config_set_backend()`.
#' ```r
#' library(keras3)
#' use_backend("tensorflow")
#' ```
#' @returns Called primarily for side effects. Returns the provided `backend`, invisibly.
#' @export
use_backend <- function(backend) {

  if (is_keras_loaded()) {
    if (config_backend() != backend)
      stop("The keras backend must be set before keras has inititialized. Please restart the R session.")
  }
  Sys.setenv(KERAS_BACKEND = backend)

  if (reticulate::py_available())
    reticulate::import("os")$environ$update(list(KERAS_BACKEND = backend))
  invisible(backend)
}


is_keras_loaded <- function() {
  # package .onLoad() has run (can be FALSE if in devtools::load_all())
  !is.null(keras) &&

  # python is initialized
  reticulate::py_available() &&

  # the keras module proxy has been resolved
  # (reticulate:::py_module_proxy_import()
  #  removes 'module' from the lazy_loaded PyObjectRef module env)
  !exists("module", envir = keras)
}



get_cudnn_path <- function(python) {

  # For TF 2.13, this assumes that someone already has cudn 11-8 installed,
  # e.g., on ubuntu:
  # sudo apt install cuda-toolkit-11-8
  # also, that `python -m pip install 'nvidia-cudnn-cu11==8.6.*'`

  force(python)
  cudnn_module_path <- suppressWarnings(system2(
    python, c("-c", shQuote("import nvidia.cudnn;print(nvidia.cudnn.__file__)")),
    stdout = TRUE, stderr = TRUE))
  if (!is.null(attr(cudnn_module_path, "status")) ||
      !is_string(cudnn_module_path) ||
      !file.exists(cudnn_module_path))
    return()

  dirname(cudnn_module_path)

}

configure_cudnn_symlinks <- function(envname) {
  python <- reticulate::virtualenv_python(envname)

  cudnn_path <- get_cudnn_path(python)
  # "~/.virtualenvs/r-keras/lib/python3.11/site-packages/nvidia/cudnn"

  cudnn_sos <- Sys.glob(paste0(cudnn_path, "/lib/*.so*"))
  # [1] "~/.virtualenvs/r-keras/lib/python3.11/site-packages/nvidia/cudnn/lib/libcudnn_adv_infer.so.8"
  # [2] "~/.virtualenvs/r-keras/lib/python3.11/site-packages/nvidia/cudnn/lib/libcudnn_adv_train.so.8"
  # [3] "~/.virtualenvs/r-keras/lib/python3.11/site-packages/nvidia/cudnn/lib/libcudnn_cnn_infer.so.8"
  # [4] "~/.virtualenvs/r-keras/lib/python3.11/site-packages/nvidia/cudnn/lib/libcudnn_cnn_train.so.8"
  # [5] "~/.virtualenvs/r-keras/lib/python3.11/site-packages/nvidia/cudnn/lib/libcudnn_ops_infer.so.8"
  # [6] "~/.virtualenvs/r-keras/lib/python3.11/site-packages/nvidia/cudnn/lib/libcudnn_ops_train.so.8"
  # [7] "~/.virtualenvs/r-keras/lib/python3.11/site-packages/nvidia/cudnn/lib/libcudnn.so.8"

  # "/home/tomasz/.virtualenvs/r-tensorflow/lib/python3.8/site-packages/tensorflow/__init__.py"
  tf_lib_path <- system2(python, c("-c", shQuote("import tensorflow as tf; print(tf.__file__)")),
                         stderr = FALSE, stdout = TRUE)
  tf_lib_path <- dirname(tf_lib_path)

  from <- sub("^.*/site-packages/", "../", cudnn_sos)
  to <- file.path(tf_lib_path, basename(cudnn_sos))
  writeLines("creating symlinks:")
  writeLines(paste("-", shQuote(to), "->", shQuote(from)))
# creating symlinks:
# - '~/.virtualenvs/r-keras/lib/python3.11/site-packages/tensorflow/libcudnn_adv_infer.so.8' -> '../nvidia/cudnn/lib/libcudnn_adv_infer.so.8'
# - '~/.virtualenvs/r-keras/lib/python3.11/site-packages/tensorflow/libcudnn_adv_train.so.8' -> '../nvidia/cudnn/lib/libcudnn_adv_train.so.8'
# - '~/.virtualenvs/r-keras/lib/python3.11/site-packages/tensorflow/libcudnn_cnn_infer.so.8' -> '../nvidia/cudnn/lib/libcudnn_cnn_infer.so.8'
# - '~/.virtualenvs/r-keras/lib/python3.11/site-packages/tensorflow/libcudnn_cnn_train.so.8' -> '../nvidia/cudnn/lib/libcudnn_cnn_train.so.8'
# - '~/.virtualenvs/r-keras/lib/python3.11/site-packages/tensorflow/libcudnn_ops_infer.so.8' -> '../nvidia/cudnn/lib/libcudnn_ops_infer.so.8'
# - '~/.virtualenvs/r-keras/lib/python3.11/site-packages/tensorflow/libcudnn_ops_train.so.8' -> '../nvidia/cudnn/lib/libcudnn_ops_train.so.8'
# - '~/.virtualenvs/r-keras/lib/python3.11/site-packages/tensorflow/libcudnn.so.8' -> '../nvidia/cudnn/lib/libcudnn.so.8'
  file.symlink(from = from, to = to)

}

