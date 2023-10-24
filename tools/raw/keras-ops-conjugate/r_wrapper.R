#' Returns the complex conjugate, element-wise.
#'
#' @description
#' The complex conjugate of a complex number is obtained by changing the sign
#' of its imaginary part.
#'
#' `keras.ops.conj` is a shorthand for this function.
#'
#' # Returns
#'     The complex conjugate of each element in `x`.
#'
#' @param x Input tensor.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/conjugate>
k_conjugate <-
function (x)
keras$ops$conjugate(x)
