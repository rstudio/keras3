#' Return the base 10 logarithm of the input tensor, element-wise.
#'
#' @description
#'
#' # Returns
#'     Output tensor, element-wise base 10 logarithm of `x`.
#'
#' @param x Input tensor.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/log10>
k_log_10 <-
function (x)
keras$ops$log10(x)
