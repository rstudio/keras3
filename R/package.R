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
#'   [Jax](https://github.com/google/jax),
#'   or [PyTorch](https://github.com/pytorch/pytorch).
#'
#' See the package website at <https://keras.rstudio.com> for complete documentation.
#'
#' @import R6
#' @importFrom reticulate
#'   import import_from_path py_install
#'   dict tuple
#'   iterate py_iterator iter_next
#'   py_call py_eval
#'   py_capture_output py_is_null_xptr
#'   py_get_attr py_has_attr
#'   py_to_r r_to_py
#'   np_array
#' @importFrom graphics par plot points
#' @importFrom tensorflow tf_version tf_config install_tensorflow all_dims
#' @aliases keras-package
"_PACKAGE"

# @import methods

# package level global state
.globals <- new.env(parent = emptyenv())



#' Main Keras module
#'
#' The `keras` module object is the equivalent of
#' `retirculate::import("keras")` and provided mainly as a convenience.
#'
#' @return the keras Python module
#' @export
#' @usage NULL
#' @format An object of class `python.builtin.module`
keras <- NULL

.onLoad <- function(libname, pkgname) {

  # if KERAS_PYTHON is defined then forward it to RETICULATE_PYTHON
  keras_python <- get_keras_python()
  if (!is.null(keras_python))
    Sys.setenv(RETICULATE_PYTHON = keras_python)

  # delay load keras
  try(keras <<- import("keras", delay_load = list(

    priority = 10, # tensorflow priority == 5

    environment = "r-keras",

    # get_module = function() {
    #   resolve_implementation_module()
    # },

    on_load = function() {
      # check version
      check_implementation_version()

      # if(implementation_module != "keras_core") {
      # if(!py_has_attr(keras, "ops"))
      #   reticulate::py_set_attr(keras, "ops",  keras$backend)

      tryCatch(
        import("tensorflow")$experimental$numpy$experimental_enable_numpy_behavior(),
        error = function(e) {
          warning("failed setting experimental_enable_numpy_behavior")
        })

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
  )))

  # register class filter to alias classes to 'keras'
  # reticulate::register_class_filter(function(classes) {
  #
  #   module <- resolve_implementation_module()
  #
  #   if (identical(module, "tensorflow.keras"))
  #     module <- "tensorflow.python.keras"
  #
  #   # replace "tensorflow.python.keras.*" with "keras.*"
  #   classes <- sub(paste0("^", module), "keras", classes)
  #
  #   # All python symbols moved in v2.13 under .src
  #   classes <- sub("^keras\\.src\\.", "keras.", classes)
  #
  #   # let KerasTensor inherit all the S3 methods of tf.Tensor, but
  #   # KerasTensor methods take precedence.
  #   if(any("keras.engine.keras_tensor.KerasTensor" %in% classes))
  #     classes <- unique(c("keras.engine.keras_tensor.KerasTensor",
  #                         "tensorflow.tensor",
  #                         classes))
  #   classes
  # })

  # tensorflow use_session hooks
  setHook("tensorflow.on_before_use_session", tensorflow_on_before_use_session)
  setHook("tensorflow.on_use_session", tensorflow_on_use_session)

  on_load_make_as_activation()

  # TODO: remove this requireNamespace()
  # temporarily here to enable passing of tests -
  # tensorflow:::.onLoad() registers some reticulate class filter hooks
  # we need to identify tensors reliably.
  requireNamespace("tensorflow")

}

keras_not_found_message <- function(error_message) {
  message(error_message)
  message("Use the install_keras() function to install the core Keras library")
}



resolve_implementation_module <- function() {

  # determine implementation to use
  module <- get_keras_implementation()

  # set the implementation module
  if (identical(module, "tensorflow"))
    module <- "tensorflow.keras"

  # return implementation_module
  module
}

get_keras_implementation <- function(default = "keras") {
  get_keras_option("KERAS_IMPLEMENTATION", default = default)
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
    update_with <- "keras3::install_keras()"
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
  if(keras$`__name__` == "keras_core")
    return(package_version("3.0"))
  ver <-
    as_r_value(py_get_attr(keras, "__version__", TRUE)) %||%
    tensorflow::tf_config()$version_str
  ver <- gsub("[^0-9.-]+", ".", as.character(ver), perl = TRUE)
  ver <- gsub("[.-]+", ".", ver, perl = TRUE)
  package_version(ver)
}



#' Check if Keras is Available
#'
#' Probe to see whether the Keras Python package is available in the current
#' system environment.
#'
#' @param version Minimum required version of Keras (defaults to `NULL`, no
#'   required version).
#'
#' @return Logical indicating whether Keras (or the specified minimum version of
#'   Keras) is available.
#'
#' @examples
#' \dontrun{
#' # testthat utilty for skipping tests when Keras isn't available
#' skip_if_no_keras <- function(version = NULL) {
#'   if (!is_keras_available(version))
#'     skip("Required keras version not available for testing")
#' }
#'
#' # use the function within a test
#' test_that("keras function works correctly", {
#'   skip_if_no_keras()
#'   # test code here
#' })
#' }
#'
#' @export
is_keras_available <- function(version = NULL) {
  implementation_module <- resolve_implementation_module()
  if (reticulate::py_module_available(implementation_module)) {
    if (!is.null(version))
      keras_version() >= version
    else
      TRUE
  } else {
    FALSE
  }
}


#' Keras implementation
#'
#' Obtain a reference to the Python module used for the implementation of Keras.
#'
#' These are the available Python modules which implement Keras:
#'
#' - keras
#' - tensorflow.keras ("tensorflow")
#' - keras_core ("core")
#'
#' This function returns a reference to the implementation being currently
#' used by the keras package. The default implementation is "keras".
#' You can override this by setting the `KERAS_IMPLEMENTATION` environment
#' variable to "tensorflow".
#'
#' @return Reference to the Python module used for the implementation of Keras.
#'
#' @export
implementation <- function() {
  keras
}
