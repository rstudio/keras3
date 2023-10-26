#' Subtract arguments element-wise.
#'
#' @description
#'
#' # Returns
#'     Output tensor, element-wise difference of `x1` and `x2`.
#'
#' @param x1 First input tensor.
#' @param x2 Second input tensor.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/subtract>
k_subtract <-
function (x1, x2)
keras$ops$subtract(x1, x2)
