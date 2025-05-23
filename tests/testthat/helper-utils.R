# Sys.setenv(TF_CPP_MIN_LOG_LEVEL = 1)
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed



reticulate:::py_register_load_hook("keras", function() {
  # browser()
  print(reticulate::py_config())
  # print(keras$`__version__`)
  # print(keras$`__path__`)
  # print(keras)

  reticulate::py_run_string(glue::trim(r"---(
    import keras
    keras.config.disable_traceback_filtering()
    )---"))

  try(reticulate::py_run_string(local = TRUE, glue::trim(r"---(
    from importlib import import_module
    import tensorflow as tf

    tf.function(lambda x: x + x)(1)

    m = import_module(tf.function.__module__)
    m.FREQUENT_TRACING_WARNING_THRESHOLD = float("inf")

    )---")))

  # py_main <- reticulate::import("__main__")
  # keras$layers # force load
  # py_main$keras <- keras
  # py_eval("keras.config.disable_traceback_filtering()")
})


# if (reticulate::py_module_available("tensorflow")) {
#   # force verbose tf init messages early
#   tensorflow::tf$`function`(function(x) tensorflow::tf$abs(x))(-1)
#
# } else
#   message("TensorFlow not available for testing")

tf_version <- tensorflow::tf_version

skip_if_no_keras <- function(required_version = NULL) {
  if (!is_keras_available(required_version))
    skip("required keras version not available for testing")
}

expect_warning_if <- function(cond, expr) {
  expect_warning(
    expr,
    regexp = if (cond) NULL else NA
  )
}

py_capture_output <- reticulate::py_capture_output #import("IPython")$utils$capture$capture_output

defer_parent <- withr::defer_parent

local_py_capture_output <- function(type = c("stdout", "stderr")) {
  stopifnot(reticulate::py_available(TRUE))
  type <- match.arg(type, several.ok = TRUE)
  output_tools <- import("rpytools.output")
  capture_stdout <- "stdout" %in% type
  capture_stderr <- "stderr" %in% type
  context_manager <- output_tools$OutputCaptureContext(
    capture_stdout, capture_stderr
  )
  context_manager$`__enter__`()

  defer_parent({
    context_manager$`__exit__`(NULL, NULL, NULL)
    context_manager$collect_output()
  })
}

local_output_sink <- withr::local_output_sink

test_succeeds <- function(desc, expr, required_version = NULL) {
  if(!is.null(required_version))
    skip_if_no_keras(required_version)


  if(!interactive()) {
    local_py_capture_output()
    local_output_sink(nullfile())
  }


  rlang::eval_tidy(
    rlang::expr(test_that(!!desc, {
      expect_no_error( {{expr}} )
      expect_true(TRUE) # https://github.com/r-lib/testthat/issues/2037
      } ))
    # , # env = parent.frame()
  )

}

test_call_succeeds <- function(call_name, expr, required_version = NULL) {
  test_succeeds(paste(call_name, "call succeeds"), expr, required_version)
}

is_backend <- function(name) {
  if (keras_version() >= "3.0")
    backend <- keras$config$backend()
  else
    backend <- backend()$backend()
  is_keras_available() && identical(backend, name)
}

skip_if_cntk <- function() {
  if (is_backend("cntk"))
    skip("Test not run for CNTK backend")
}

skip_if_theano <- function() {
  if (is_backend("theano"))
    skip("Test not run for theano backend")
}

skip_if_tensorflow_implementation <- function() {
  if (keras3:::is_tensorflow_implementation())
    skip("Test not run for TensorFlow implementation")
}

define_model <- function() {
  model <- keras_model_sequential(input_shape = 784)
  model %>%
    layer_dense(32, kernel_initializer = initializer_ones()) %>%
    layer_activation('relu') %>%
    layer_dense(10) %>%
    layer_activation('softmax')
  model
}

define_and_compile_model <- function() {
  model <- define_model()
  model %>%
    compile(
      loss='binary_crossentropy',
      optimizer = optimizer_sgd(),
      metrics='accuracy'
    )
  model
}


expect_tensor <- function(x, shape=NULL, shaped_as=NULL) {
  x_lbl <- quasi_label(rlang::enquo(x), arg = 'x')$lab

  expect(op_is_tensor(x) ||
           keras$backend$is_keras_tensor(x) ||
           inherits(x, "tensorflow.tensor"),
         paste(x_lbl, "was wrong S3 class, expected a tensor, actual", class(x)))

  x_shape <- x$shape

  if(!is.list(x_shape)) # tensorflow TensorShape()
    x_shape <- x_shape$as_list()

  x_shape <- as.list(x_shape)

  chk_expr <- quote(expect(
    identical(x_shape, shape),
    sprintf("%s was wrong shape, expected: %s, actual: %s",
            x_lbl, x_shape, shape)
  ))

  if(!is.null(shape)) {
    shape <- as.list(shape)
    eval(chk_expr)
  }

  if(!is.null(shaped_as)) {
    shape <- shaped_as$shape
    if(!is.list(shape))
      shape <- shape$as_list()
    eval(chk_expr)
  }
  invisible(x)
}


expect_same_pyobj <- function(x, y) {
  eval.parent(bquote(expect_identical(
    get0("pyobj", as.environment(.(x))),
    get0("pyobj", as.environment(.(y)))
  )))
}


repl_python <- reticulate::repl_python
py_last_error <- reticulate::py_last_error
iter_next <- reticulate::iter_next
as_iterator <- reticulate::as_iterator

tf <- tensorflow::tf
as_tensor <- tensorflow::as_tensor

# modeled after withr::local_
local_tf_device <- function(device_name = "CPU") {
  device <- tf$device(device_name)
  device$`__enter__`()
  withr::defer_parent(device$`__exit__`())
  invisible(device)
}

k_constant <- function(value, dtype = NULL, shape = NULL, name = NULL) {
  if(!is.null(name)) stop("no name")
  x <- reticulate::np_array(value, dtype)
  if(!is.null(shape))
    x <- x$reshape(as.integer(shape))
  x
  keras$ops$convert_to_tensor(x)
}


ct <- function(...) {
  x <- if(...length() == 1L) ..1 else c(...)
  if (is.double(x))
    storage.mode(x) <- "integer"
  op_convert_to_tensor(as.array(x))
}

as.tensor <- function(x) op_convert_to_tensor(x)

env <- rlang::env


expect_equal_array <- function(x, y) {
  expect_equal(as.array(x), as.array(y))
}
