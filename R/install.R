#' Install Keras
#'
#' This function will install Keras along with a selected backend, including all Python dependencies.
#'
#' @param envname Name of or path to a Python virtual environment
#' @param extra_packages Additional Python packages to install alongside Keras
#' @param python_version Passed on to `reticulate::virtualenv_starter()`
#' @param backend Which backend. Accepted values include  `"tensorflow"`, `"jax"` and `"pytorch"`
#' @param gpu whether to install a GPU capable version of the backend.
#' @param ... reserved for future compatability.
#'
#' @seealso [`tensorflow::install_tensorflow()`]
#' @export
install_keras <- function(
    envname = "r-keras", ...,
    extra_packages = c("scipy", "pandas", "Pillow", "pydot", "ipython"),
    python_version = ">=3.9,<=3.11",
    backend = "tf-nightly", # c("tensorflow", "jax"),
    gpu = NA,
    restart_session = TRUE) {

  if(is.na(gpu)) {

    has_nvidia_gpu <- function()
      tryCatch(as.logical(length(system("lspci | grep -i nvidia", intern = TRUE))),
               warning = function(w) FALSE)
    gpu <- (is_linux() && has_nvidia_gpu()) ||
            is_mac_arm64()
  }

  if (isTRUE(gpu)) {
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
    # undeclared dependancy, import fails otherwise
    append(extra_packages) <- "packaging"

  backend <- unlist(lapply(backend, function(name)
    switch(name, tensorflow = tensorflow, jax = jax, name)))

  reticulate::virtualenv_create(
    envname = envname,
    version = python_version,
    force = identical(envname, "r-keras"),
    packages = NULL
  )

  if (length(extra_packages))
    reticulate::py_install(unique(extra_packages), envname = envname)

  if (length(backend))
    reticulate::py_install(backend, envname = envname)

  reticulate::py_install("keras==3.0.*", envname = envname)

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
