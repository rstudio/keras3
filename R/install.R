#' Install Keras
#'
#' This function will install Keras along with a selected backend, including all Python dependencies.
#'
#' @param envname Name of or path to a Python virtual environment
#' @param extra_packages Additional Python packages to install alongside Keras
#' @param python_version Passed on to `reticulate::virtualenv_starter()`
#' @param backend Which backend. Accepted values include  `"tensorflow"`, `"jax"` and `"pytorch"`
#' @param ... reserved for future compatability.
#'
#' @seealso [`tensorflow::install_tensorflow()`]
#' @export
install_keras <- function(
    envname = "r-keras", ...,
    extra_packages = c("scipy", "pandas", "Pillow", "pydot", "ipython"),
    python_version = ">=3.9,<=3.11",
    backend = "tf-nightly",
    # cuda = NA # infer GPU, customize backend like "tensorflow[and-cuda]" or "jax[cuda]"
    restart_session = TRUE) {

  reticulate::virtualenv_create(
    envname = envname,
    version = python_version,
    force = identical(envname, "r-keras"),
    packages = backend
  )

  reticulate::py_install(extra_packages, envname = envname)
  reticulate::py_install("keras==3.0.*", envname = envname)

  message("Finished installing Keras!")
  if (restart_session && requireNamespace("rstudioapi", quietly = TRUE) &&
    rstudioapi::hasFun("restartSession")) {
    rstudioapi::restartSession()
  }

  invisible(NULL)
}

#' @export
#' @rdname config_backend
#' @param value string, one of `"tensorflow"`, `"jax"`, or `"torch"`.
config_set_backend <- function(value = c("tensorflow", "jax", "torch")) {
  value <- match.arg(value)
  py_inited <- reticulate::py_available()

  # is the keras module already imported? reticulate:::py_module_proxy_import()
  # removes 'module' from the lazy_loaded PyObjectRef module env
  keras_module_resolved <- !exists("module", envir = keras)

  if(py_inited && keras_module_resolved) {
    if(config_backend() == value)
      return(invisible(value))
    else
      stop("The keras backend must be set before keras has inititialized. Please restart the R session.")
  }
  Sys.setenv(KERAS_BACKEND = value)
  if(py_inited)
    reticulate::import("os")$environ$update(list(KERAS_BACKEND = value))
  invisible(value)
}
