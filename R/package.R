#' R interface to Keras
#'
#' Keras is a high-level neural networks API, developed with a focus on enabling
#' fast experimentation. Keras has the following key features:
#'
#' - Allows the same code to run on CPU or on GPU, seamlessly.
#' - User-friendly API which makes it easy to quickly prototype deep learning models.
#' - Built-in support for convolutional networks (for computer vision), recurrent
#'   networks (for sequence processing), and any combination of both.
#' - Supports arbitrary network architectures: multi-input or multi-output models,
#'   layer sharing, model sharing, etc. This means that Keras is appropriate for
#'   building essentially any deep learning model, from a memory network to a neural
#'   Turing machine.
#' - Is capable of running on top of multiple back-ends including
#'   [TensorFlow](https://github.com/tensorflow/tensorflow),
#'   [CNTK](https://github.com/Microsoft/cntk),
#'   or [Theano](https://github.com/Theano/Theano).
#'
#' See the package website at <https://keras.rstudio.com> for complete documentation.
#'
#' @import methods
#' @import R6
#' @importFrom reticulate import dict iterate import_from_path py_iterator py_call py_capture_output py_get_attr py_has_attr py_is_null_xptr py_to_r r_to_py tuple
#' @importFrom graphics par plot points
#' @importFrom tensorflow tf_version tf_config install_tensorflow
#' @aliases keras-package
"_PACKAGE"

# package level global state
.globals <- new.env(parent = emptyenv())


#' Select a Keras implementation and backend
#'
#' @param implementation One of "keras" or "tensorflow" (defaults to "keras").
#' @param backend One of "tensorflow", "cntk", or "theano" (defaults
#'   to "tensorflow")
#'
#' @details
#' Keras has multiple implementations (the original keras implementation
#' and the implementation native to TensorFlow) and supports multiple
#' backends ("tensorflow", "cntk", "theano", and "plaidml"). These functions allow
#' switching between the various implementations and backends.
#'
#' The functions should be called after `library(keras)` and before calling
#' other functions within the package (see below for an example).
#'
#' The default implementation and backend should be suitable for most
#' use cases. The "tensorflow" implementation is useful when using Keras
#' in conjunction with TensorFlow Estimators (the \pkg{tfestimators}
#' R package).
#'
#' @examples \dontrun{
#' # use the tensorflow implementation
#' library(keras)
#' use_implementation("tensorflow")
#'
#' # use the cntk backend
#' library(keras)
#' use_backend("theano")
#' }
#'
#' @export
use_implementation <- function(implementation = c("keras", "tensorflow")) {
  Sys.setenv(KERAS_IMPLEMENTATION = match.arg(implementation))
}


#' @rdname use_implementation
#' @export
use_backend <- function(backend = c("tensorflow", "cntk", "theano", "plaidml")) {
  backend <- match.arg(backend)
  if (backend == "plaidml") {
    pml_keras <- import("plaidml.keras", delay_load = list(
      priority = 20
    ))
    pml_keras$install_backend()
  } else {
    Sys.setenv(KERAS_BACKEND = match.arg(backend))
  }
  if (backend != "tensorflow") use_implementation("keras")
}



#' Main Keras module
#'
#'
#' The `keras` module object is the equivalent of
#' `keras <- tensorflow::tf$keras` and provided mainly as a convenience.
#'
#' @return the keras Python module
#' @export
keras <- NULL


.onLoad <- function(libname, pkgname) {

  # resolve the implementation module (might be keras proper or might be tensorflow)
  implementation_module <- resolve_implementation_module()

  # if KERAS_PYTHON is defined then forward it to RETICULATE_PYTHON
  keras_python <- get_keras_python()
  if (!is.null(keras_python))
    Sys.setenv(RETICULATE_PYTHON = keras_python)

  # delay load keras
  keras <<- import(implementation_module, delay_load = list(

    priority = 10,

    environment = "r-reticulate",

    get_module = function() {
      resolve_implementation_module()
    },

    on_load = function() {
      # check version
      check_implementation_version()

      # patch progress bar for interactive/tty sessions
      if ((interactive() || isatty(stdout())) && keras_version() >= "2.0.9") {
        python_path <- system.file("python", package = "keras")
        tools <- import_from_path("kerastools", path = python_path)
        tools$progbar$apply_patch()
      }

    },

    on_error = function(e) {
      if (is_tensorflow_implementation())
        stop(tf_config()$error_message, call. = FALSE)
      else {
        if (grepl("No module named keras", e$message)) {
          keras_not_found_message(e$message)
        } else {
          stop(e$message, call. = FALSE)
        }
      }
    }
  ))

  # register class filter to alias classes to 'keras'
  reticulate::register_class_filter(function(classes) {

    module <- resolve_implementation_module()

    if (identical(module, "tensorflow.keras"))
      module <- "tensorflow.python.keras"

    classes <- sub(paste0("^", module), "keras", classes)


    # let KerasTensor inherit all the S3 methods of tf.Tensor, but
    # KerasTensor methods take precedence.
    if("keras.engine.keras_tensor.KerasTensor" %in% classes)
      classes <- unique(c("keras.engine.keras_tensor.KerasTensor",
                          "tensorflow.tensor",
                          classes))
    classes
  })

  # tensorflow use_session hooks
  setHook("tensorflow.on_before_use_session", tensorflow_on_before_use_session)
  setHook("tensorflow.on_use_session", tensorflow_on_use_session)

  registerS3method("[[", "python_class_super", active_extract2)
  registerS3method("$", "python_class_super", active_extract2)

}

keras_not_found_message <- function(error_message) {
  message(error_message)
  message("Use the install_keras() function to install the core Keras library")
}

active_extract2 <- function(x, name) {
  name <- switch(name,
                 "initialize" = "__init__",
                 "finalize" = "__del__",
                 name)
  cl <- substitute(x()$name,  list(x = substitute(x), name = as.symbol(name)))
  eval.parent(cl)
}



resolve_implementation_module <- function() {

  # determine implementation to use
  implementation <- get_keras_implementation()

  # set the implementation module
  if (identical(implementation, "tensorflow"))
    implementation_module <- "tensorflow.keras"
  else
    implementation_module <- implementation

  # return implementation_module
  implementation_module
}

get_keras_implementation <- function(default = "tensorflow") {
  out <- get_keras_option("KERAS_IMPLEMENTATION", default = default)
  if (out != "tensorflow")
    rlang::warn(c(
      paste0("We no longer support the '", out, "' Keras implementation."),
      "Use Sys.setenv(KERAS_IMPLEMENTATION='tensorflow') or unset that environment variable."
    ), .frequency = "once", .frequency_id = "implementation")
  out
}

get_keras_python <- function(default = NULL) {
  get_keras_option("KERAS_PYTHON", default = default, as_lower = FALSE)
}

get_keras_option <- function(name, default = NULL, as_lower = TRUE) {

  # case helper
  uncase <- function(x) {
    if (as_lower && !is.null(x) && !is.na(x))
      tolower(x)
    else
      x
  }

  value <- Sys.getenv(name, unset = NA)
  if (!is.na(value))
    uncase(value)
  else
    uncase(default)
}


is_tensorflow_implementation <- function(implementation = get_keras_implementation()) {
  grepl("^tensorflow", implementation)
}

is_keras_implementation <- function(implementation = get_keras_implementation()) {
  identical(implementation, "keras")
}

check_implementation_version <- function() {

  # get current implementation
  implementation <- get_keras_implementation()

  # version variables
  ver <- NULL
  required_ver <- NULL

  # define implemetation-specific version/required-version
  if (is_tensorflow_implementation(implementation)) {
    name <- "TensorFlow"
    ver <- tf_version()
    required_ver <- "1.9"
    update_with <- "tensorflow::install_tensorflow()"
  } else if (is_keras_implementation(implementation)) {
    name <- "Keras"
    ver <- keras_version()
    required_ver <- "2.0.0"
    update_with <- "keras::install_keras()"
  }

  # check version if we can
  if (!is.null(required_ver)) {
    if (ver < required_ver) {
      stop("Keras loaded from ", implementation, " v", ver, ", however version ",
            required_ver, " is required. Please update with ", update_with, ".",
           call. = FALSE)
    }
  }
}


# Current version of Keras
keras_version <- function() {
  ver <- keras$`__version__`
  ver <- regmatches(ver, regexec("^([0-9\\.]+).*$", ver))[[1]][[2]]
  package_version(ver)
}
