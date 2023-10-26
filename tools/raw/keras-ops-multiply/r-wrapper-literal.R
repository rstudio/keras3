#' Multiply arguments element-wise.
#'
#' @description
#'
#' # Returns
#'     Output tensor, element-wise product of `x1` and `x2`.
#'
#' @param x1 First input tensor.
#' @param x2 Second input tensor.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/multiply>
k_multiply <-
function (x1, x2)
keras$ops$multiply(x1, x2)
