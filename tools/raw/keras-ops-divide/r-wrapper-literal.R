#' Divide arguments element-wise.
#'
#' @description
#' `keras.ops.true_divide` is an alias for this function.
#'
#' # Returns
#'     Output tensor, the quotient `x1/x2`, element-wise.
#'
#' @param x1 First input tensor.
#' @param x2 Second input tensor.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/divide>
k_divide <-
function (x1, x2)
keras$ops$divide(x1, x2)
