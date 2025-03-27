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
#'   [Jax](https://github.com/jax-ml/jax),
#'   or [PyTorch](https://github.com/pytorch/pytorch).
#'
#' See the package website at <https://keras3.posit.co> for complete documentation.
#'
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
#' @aliases keras3-package
"_PACKAGE"


# package level global state
.globals <- new.env(parent = emptyenv())

tf <- NULL
ops <- NULL
np <- NULL


#' Main Keras module
#'
#' The `keras` module object is the equivalent of
#' `reticulate::import("keras")` and provided mainly as a convenience.
#'
#' @returns the keras Python module
#' @export
#' @usage NULL
#' @format An object of class `python.builtin.module`
keras <- NULL

.onLoad <- function(libname, pkgname) {

  if (is.na(Sys.getenv("TF_CPP_MIN_LOG_LEVEL", NA)))
    Sys.setenv("TF_CPP_MIN_LOG_LEVEL" = "2")

  # tensorflow:::.onLoad() registers some reticulate class filter hooks
  # we need to identify tensorflow tensors reliably.
  requireNamespace("tensorflow", quietly = TRUE)
  maybe_register_S3_methods()

  registerS3method("%*%", "tensorflow.tensor", op_matmul, baseenv())

  # if KERAS_PYTHON is defined then forward it to RETICULATE_PYTHON
  keras_python <- get_keras_python()
  if (!is.null(keras_python))
    Sys.setenv(RETICULATE_PYTHON = keras_python)

  py_require(c(
    "keras", "pydot", "scipy", "pandas", "Pillow",
    "ipython" #, "tensorflow_datasets"
  ))

  if (is.na(Sys.getenv("KERAS_HOME", NA))) {
    if (!dir.exists("~/.keras/")) {
      Sys.setenv("KERAS_HOME" = normalizePath(
        tools::R_user_dir("keras3", "cache"),
        mustWork = FALSE
      ))
    }
  }

  # default backend is tensorflow for now
  # the tensorflow R package calls `py_require()` to ensure GPU is usable on Linux
  # use_backend() includes py_require(action = "remove") calls to undo
  # what tensorflow:::.onLoad() did. Keep them in sync!
  # backend <- Sys.getenv("KERAS_BACKEND", "jax")
  # ~/.keras.keras.json also has an (undocumented) 'backend' field
  backend <- Sys.getenv("KERAS_BACKEND", "tensorflow")
  gpu <- NA
  if (endsWith(backend, "-cpu")) {
    gpu <- FALSE
    backend <- sub("-cpu$", "", backend)
    Sys.setenv("KERAS_BACKEND" = backend)
  } else if (endsWith(backend, "-gpu")) {
    gpu <- TRUE
    backend <- sub("-gpu$", "", backend)
    Sys.setenv("KERAS_BACKEND" = backend)
  }

  if(Sys.getenv("DEVTOOLS_LOAD") == "keras3") {
    if (Sys.getenv("KERAS_BACKEND_CONFIGURED") != "yes") {
      use_backend(backend, gpu)
      Sys.setenv("KERAS_BACKEND_CONFIGURED" = "yes")
    }
  } else {
    use_backend(backend, gpu)
  }


  # delay load keras
  try(keras <<- import("keras", delay_load = list(

    priority = 10, # tensorflow priority == 5

    environment = "r-keras",

    # get_module = function() {
    #   resolve_implementation_module()
    # },

    on_load = function() {
      # check version
      # check_implementation_version()

      # disabled because of errors with keras-hub
      # tryCatch(
      #   import("tensorflow")$experimental$numpy$experimental_enable_numpy_behavior(),
      #   error = function(e) {
      #     warning("failed setting experimental_enable_numpy_behavior")
      #   })

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

  reticulate::py_register_load_hook("keras", function() {

    keras <- import("keras")
    convert_to_tensor <- import("keras.ops", convert = FALSE)$convert_to_tensor
    with(keras$device("cpu:0"), {
      backend_tensor_class <- class(convert_to_tensor(array(1L)))[1L]
    })
    symbolic_tensor_class <- nameOfClass__python.builtin.type(keras$KerasTensor)

    registerS3method("@", symbolic_tensor_class, at.keras_backend_tensor, baseenv())
    registerS3method("@", backend_tensor_class, at.keras_backend_tensor, baseenv())
    registerS3method("@", "numpy.ndarray", at.keras_backend_tensor, baseenv())

    py_subset <- utils::getS3method("[", "python.builtin.object", envir = asNamespace("reticulate"))
    registerS3method("[", "keras_r_backend_tensor", op_subset, baseenv())
    registerS3method("[", "keras_py_backend_tensor", py_subset, baseenv())

    registerS3method("@<-", symbolic_tensor_class, at_set.keras_backend_tensor, baseenv())
    registerS3method("@<-", backend_tensor_class, at_set.keras_backend_tensor, baseenv())
    registerS3method("@<-", "numpy.ndarray", at_set.keras_backend_tensor, baseenv())

    `py_subset<-` <- utils::getS3method("[<-", "python.builtin.object", envir = asNamespace("reticulate"))
    registerS3method("[<-", "keras_r_backend_tensor", `op_subset<-`, baseenv())
    registerS3method("[<-", "keras_py_backend_tensor", `py_subset<-`, baseenv())

    registerS3method("as.array", backend_tensor_class, op_convert_to_array, baseenv())
    registerS3method("^", backend_tensor_class, `^__keras.backend.tensor`, baseenv())
    registerS3method("%*%", backend_tensor_class, op_matmul, baseenv())

  })



  reticulate::py_register_load_hook("torch", function() {
    # force keras load hooks to run
    keras$ops
  })

  reticulate::py_register_load_hook("jax", function() {
    # force keras load hooks to run
    keras$ops
  })


  reticulate::py_register_load_hook("tensorflow", function() {

    # Globally enabling this is too disruptive - causes
    # errors in tf internal calls like `tf.strings.split("foo\nbar", "\n")`
    # also, internal keras calls in `fit()` that check for overflow.
    # we only use numpy style slicing via an internal method in op_subset()

    # tf <- import("tensorflow")
    # if(Sys.getenv("TENSORFLOW_ENABLE_NUMPY_BEHAVIOR") != "false")
    # py_capture_output({
    #   tf$experimental$numpy$experimental_enable_numpy_behavior(
    #     prefer_float32 = TRUE,
    #     dtype_conversion_mode = "legacy"
    #     # "all" or "safe" leads to error in keras
    #     # can optionally also do "off", but that's even more strict
    #     # dtype_conversion_mode = "off"
    #   )
    # }, "stderr")

    # we still need to register tensorflow `@` and `@<-` methods even if the
    # backend is not tensorflow, because tf.data can be used with other backends
    # and tensorflow.tensor might still be encountered.
    registerS3method("@", "tensorflow.tensor", at.keras_backend_tensor, baseenv())
    registerS3method("@<-", "tensorflow.tensor", at_set.keras_backend_tensor, baseenv())
  })

  # on_load_make_as_activation()
  np <<- try(import("numpy", convert = FALSE, delay_load = TRUE))
  tf <<- try(import("tensorflow", delay_load = TRUE))
  ops <<- try(import("keras.ops", delay_load = list(
    before_load = function() {
      # force the load hooks on 'keras' to run
      keras$ops
    }
  )))

}


at.keras_backend_tensor <-  function(object, name) {
  out <- rlang::env_clone(object)
  attrs <- attributes(object)
  cls <- switch(
    name,
    r = "keras_r_backend_tensor" ,
    py = "keras_py_backend_tensor",
    stop("<subset-style> must be 'r' or 'py' in expression <tensor>@<subset-style>")
  )
  attrs$class <- unique(c(cls, attrs$class))
  attributes(out) <- attrs
  out
}


at_set.keras_backend_tensor <- function(object, name, value) {
  value
}


keras_not_found_message <- function(error_message) {
  message(error_message)
  message("Use the install_keras() function to install the core Keras library")
}

maybe_register_S3_methods <- function() {
  # Tensorflow 2.16 exports these methods, but we don't need to
  # take a dep on TF>=2.16. So we conditionally export them if installed
  # tensorflow package is older. This is to avoid a warning about
  # overwritten S3 methods on package load.
  .register_no_overwrite <- function(class) {
    if (is.null(utils::getS3method("py_to_r", class, optional = TRUE,
                                   envir = asNamespace("reticulate")))) {

      # __ instead of . to avoid a roxygen warning about unexported S3 methods
      method <- get(paste0("py_to_r__", class))
      registerS3method("py_to_r", class, method,
                       envir = asNamespace("reticulate"))
    }
  }

  .register_no_overwrite("keras.src.utils.tracking.TrackedDict")
  .register_no_overwrite("keras.src.utils.tracking.TrackedList")
  .register_no_overwrite("keras.src.utils.tracking.TrackedSet")
}

# not exported regular function since nameOfClass() requires R>4.3
# __ instead of . to avoid roxygen warning
nameOfClass__python.builtin.type <- function(x) {
  paste(
    as_r_value(py_get_attr(x, "__module__")),
    as_r_value(py_get_attr(x, "__name__")),
    sep = "."
  )
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
    return(package_version("3.0.0"))
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
#' @returns Logical indicating whether Keras (or the specified minimum version of
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
#' @noRd
# @export
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

# TODO: add option in `is_keras_available()` to avoid initializing Python
#       (maybe in a callr call?), reexport.
# TODO: add func `is_backend_available()`, usage `is_backend_available("tensorflow")`


#' New axis
#'
#' This is an alias for `NULL`. It is meant to be used in `[` on tensors,
#' to expand dimensions of a tensor
#'
#' ```r
#' x <- op_convert_to_tensor(1:10)
#'
#' op_shape(x)
#' op_shape(x[])
#' op_shape(x[newaxis])
#' op_shape(x@py[newaxis])
#' op_shape(x@r[newaxis])
#'
#' op_shape(x[newaxis, .., newaxis])
#' op_shape(x@py[newaxis, .., newaxis])
#' op_shape(x@r[newaxis, .., newaxis])
#' ````
#' @export
newaxis <- NULL
