

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


k_is_tensor <- function(x) keras$utils$is_keras_tensor(x)

k_clear_session <- function() keras$backend$clear_session()

k_variable <- function(...) keras$Variable(...)

k_backend <- function() keras$config$backend()

#' @export
`==.keras.backend.common.keras_tensor.KerasTensor` <- function(e1, e2) {
  k_equal(e1, e2)
}

#' @export
`+.keras.backend.common.keras_tensor.KerasTensor` <- function(e1, e2) {
  if(missing(e2)) return(e1)
  NextMethod()
}
