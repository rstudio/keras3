#' Install Keras
#'
#' This function will install Keras along with a selected backend, including all Python dependencies.
#'
#' @param envname Name of or path to a Python virtual environment
#' @param extra_packages Additional Python packages to install alongside Keras
#' @param python_version Passed on to `reticulate::virtualenv_starter()`
#' @param backend Which backend(s) to install. Accepted values include `"tensorflow"`, `"jax"` and `"pytorch"`
#' @param gpu whether to install a GPU capable version of the backend.
#' @param restart_session Whether to restart the R session after installing (note this will only occur within RStudio).
#' @param ... reserved for future compatability.
#'
#' @seealso [`tensorflow::install_tensorflow()`]
#' @export
install_keras <- function(
    envname = "r-keras", ...,
    extra_packages = c("scipy", "pandas", "Pillow", "pydot", "ipython", "tensorflow_datasets"),
    python_version = ">=3.9,<=3.11",
    backend = "tensorflow",
    # backend = c("tensorflow", "jax"),
    # backend = "tf-nightly",
    gpu = NA,
    restart_session = TRUE) {

  if (is.na(gpu)) {
    has_nvidia_gpu <- function()
      tryCatch(as.logical(length(system("lspci | grep -i nvidia", intern = TRUE))),
               warning = function(w) FALSE)
    gpu <- (is_linux() && has_nvidia_gpu()) || is_mac_arm64()
  }

  # keras requires tensorflow be installed still.
  if(!any(grepl("tensorflow|tf-nightly", backend)))
    backend <- c("tensorflow", backend)

  if (isTRUE(gpu)) {
    message("Installing GPU components")
    if (is_mac_arm64()) {
      jax <- c("jax-metal", "ml-dtypes==0.2.*")
      tensorflow <- c("tensorflow", "tensorflow-metal")
    } else if (is_linux()) {
      jax <- c("jax[cuda12_pip]", "-f",
         "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
      tensorflow <- "tensorflow[and-cuda]"
    }
  } else {
    jax <- "jax[cpu]"
    tensorflow <- "tensorflow-cpu"
  }

  if("jax" %in% backend && !is.null(extra_packages))
    # undeclared dependency, import fails otherwise
    append(extra_packages) <- "packaging"

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

  reticulate::py_install("keras==3.0.*", envname = envname)
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
#' @param backend string, one of `"tensorflow"`, `"jax"`, or `"torch"`. Defaults to `"tensorflow"`.
#'
#' @details
#' These functions allow configuring which backend keras will use.
#' Note that only one backend can be configured per R session.
#'
#' The function should be called after `library(keras3)` and before calling
#' other functions within the package (see below for an example).
#' ```r
#' library(keras3)
#' use_backend("tensorflow")
#' ```
#' @export
use_backend <- function(backend = c("tensorflow", "jax", "torch")) {
  backend <- match.arg(backend)

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
  # package .onLoad() has run
  !is.null(keras) &&

  # python is initialized
  reticulate::py_available() &&

  # the keras module proxy has been resolved
  # (reticulate:::py_module_proxy_import()
  #  removes 'module' from the lazy_loaded PyObjectRef module env)
  !exists("module", envir = keras)
}
