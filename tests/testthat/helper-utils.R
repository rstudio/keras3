Sys.setenv(TF_CPP_MIN_LOG_LEVEL = 1)


# reticulate::use_condaenv("tf-2.5-cpu", required = TRUE)
# reticulate::use_condaenv("tf-2.1-cpu", required = TRUE)

if (reticulate::py_module_available("tensorflow")) {
  if (!exists(".DID_EMIT_TF_VERSION", envir = .GlobalEnv)) {
    message("Testing Against Tensorflow Version: ",
            tensorflow::tf$version$VERSION)
    .GlobalEnv$.DID_EMIT_TF_VERSION <- TRUE
    tensorflow::tf$`function`(function(x) tensorflow::tf$abs(x))(-1) # force tf init verbose messages early
  }
} else
  message("TensorFlow not available for testing")

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


test_succeeds <- function(desc, expr, required_version = NULL) {
  if (interactive()) {
    test_that(desc, expect_error(force(expr), NA))
  } else
    invisible(capture.output({
      test_that(desc, {
        skip_if_no_keras(required_version)
        py_capture_output({
          expect_error(force(expr), NA)
        })
      })
    }))
}

test_call_succeeds <- function(call_name, expr, required_version = NULL) {
  test_succeeds(paste(call_name, "call succeeds"), expr, required_version)
}

is_backend <- function(name) {
  is_keras_available() && identical(backend()$backend(), name)
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
  if (keras:::is_tensorflow_implementation())
    skip("Test not run for TensorFlow implementation")
}

define_model <- function() {
  model <- keras_model_sequential()
  model %>%
    layer_dense(32, input_shape = 784, kernel_initializer = initializer_ones()) %>%
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

random_array <- function(dim) {
  array(runif(prod(dim)), dim = dim)
}




expect_tensor <- function(x, shape=NULL, shaped_as=NULL) {
  x_lbl <- quasi_label(rlang::enquo(x), arg = 'x')$lab
  expect(is_keras_tensor(x),
         paste(x_lbl, "was wrong S3 class, expected 'tensorflow.tensor', actual", class(x)))

  x_shape <- x$shape$as_list()

  chk_expr <- quote(expect(
    identical(x_shape, shape),
    sprintf("%s was wrong shape, expected: %s, actual: %s", x_lbl, x_shape, shape)
  ))

  if(!is.null(shape)) {
    eval(chk_expr)
  }

  if(!is.null(shaped_as)) {
    shape <- shaped_as$shape$as_list()
    eval(chk_expr)
  }
  invisible(x)
}


expect_same_pyobj <- function(x, y) {
  eval.parent(bquote(expect_identical(
    get("pyobj", as.environment(.(x))),
    get("pyobj", as.environment(.(y)))
  )))
}


repl_python <- reticulate::repl_python
py_last_error <- reticulate::py_last_error
iter_next <- reticulate::iter_next
as_iterator <- reticulate::as_iterator

tf <- tensorflow::tf
as_tensor <- tensorflow::as_tensor
