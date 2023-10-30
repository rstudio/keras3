#' Compute the absolute value element-wise.
#'
#' @description
#' `keras::k_abs` is a shorthand for this function.
#'
#' @param x Input tensor.
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/absolute>
#'
#' @return
#' A tensor containing the absolute value of each element in `x`.
#'
#' @examples
#' x <- k_convert_to_tensor(c(-1.2, 1.2))
#' k_abs(x)
#' # array([1.2, 1.2], dtype=float32)
k_absolute <-
function (x)
keras$ops$absolute(x)
