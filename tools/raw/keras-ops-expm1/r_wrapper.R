#' Calculate `exp(x) - 1` for all elements in the tensor.
#'
#' @description
#'
#' # Returns
#'     Output tensor, element-wise exponential minus one.
#'
#' @param x Input values.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/expm1>
k_expm_1 <-
function (x)
keras$ops$expm1(x)
