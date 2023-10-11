

#' Keras backend tensor engine
#'
#' Obtain a reference to the `keras.backend` Python module used to implement
#' tensor operations.
#'
#' @inheritParams reticulate::import
#'
#' @note See the documentation here <https://keras.io/backend/> for
#'   additional details on the available functions.
#'
#' @return Reference to Keras backend python module.
#'
#' @export
backend <- function(convert = TRUE) {
  if (convert)
    keras$ops
  else
    r_to_py(keras$ops)
}


as_axis <- function(axis) {
  if (is.null(axis))
    return(NULL)

  if (length(axis) > 1)
    return(lapply(axis, as_axis))

  axis <- as.integer(axis)

  if (axis == 0L)
    stop("`axis` argument is 1 based, received 0")

  if (axis > 0L) axis - 1L
  else axis
}


backend_normalize_shape <- function(shape) {

  # if it's a Python object or a list with python objects then leave it alone
  if (inherits(shape, "python.builtin.object"))
    return(shape)

  normalize_shape(shape)
}

#' Backcompat backend ops
#'
#' @export
#' @rdname k_random_bernoulli
k_random_binomial <-
  function(shape, p = 0.0, dtype = NULL, seed = NULL) {

    tf <- import("tensorflow")
    x <- tf$random$uniform(shape = backend_normalize_shape(shape),
                           dtype = dtype %||% keras$config$floatx(),
                           seed = as_integer(seed))

    x > (1 - p)
  }

#' @export
#' @rdname k_random_bernoulli
k_random_bernoulli <- k_random_binomial


#' Returns a tensor with normal distribution of values.
#'
#' @param shape A list of integers, the shape of tensor to create.
#' @param mean A float, mean of the normal distribution to draw samples.
#' @param stddev A float, standard deviation of the normal distribution to draw
#'   samples.
#' @param dtype String, dtype of returned tensor.
#' @param seed Integer, random seed.
#'
#' @return A tensor.
#'
#' @template roxlate-keras-backend
#'
#' @export
k_random_normal <- function(shape, mean = 0.0, stddev = 1.0, dtype = NULL, seed = NULL) {
  args <- capture_args2(list(shape = backend_normalize_shape, seed = as_integer))
  tf <- import("tensorflow")
  tf$random$normal(!!!args)
}




#' Returns a tensor with uniform distribution of values.
#'
#' @param shape A list of integers, the shape of tensor to create.
#' @param minval A float, lower boundary of the uniform distribution to draw samples.
#' @param maxval A float, upper boundary of the uniform distribution to draw samples.
#' @param dtype String, dtype of returned tensor.
#' @param seed Integer, random seed.
#'
#' @return A tensor.
#'
#' @template roxlate-keras-backend
#'
#' @export
k_random_uniform <- function(shape, minval = 0.0, maxval = 1.0, dtype = NULL, seed = NULL) {
  if(!is.null(dtype)) {
    minval <- as_tensor(minval, dtype = dtype)
    maxval <- as_tensor(maxval, dtype = dtype)
  }
  args <- capture_args2(list(shape = backend_normalize_shape, seed = as_integer))
  tf <- import("tensorflow")
  tf$random$uniform(!!!args)
}

k_is_tensor <- function(x) keras$utils$is_keras_tensor(x)

k_clear_session <- function() keras$backend$clear_session()

k_variable <- function(...) keras$Variable(...)

k_backend <- function() keras$config$backend()


