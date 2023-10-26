#' Element-wise minimum of `x1` and `x2`.
#'
#' @description
#'
#' # Returns
#'     Output tensor, element-wise minimum of `x1` and `x2`.
#'
#' @param x1 First tensor.
#' @param x2 Second tensor.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/minimum>
k_minimum <-
function (x1, x2)
keras$ops$minimum(x1, x2)
