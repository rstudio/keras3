#' Natural logarithm, element-wise.
#'
#' @description
#'
#' # Returns
#'     Output tensor, element-wise natural logarithm of `x`.
#'
#' @param x Input tensor.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/log>
k_log <-
function (x)
keras$ops$log(x)
