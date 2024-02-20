#' Safe element-wise division which returns 0 where the denominator is 0.
#'
#' @returns
#' The quotient `x1/x2`, element-wise, with zero where x2 is zero.
#'
#' @param x1
#' First input tensor.
#'
#' @param x2
#' Second input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.divide_no_nan
# @seealso
# + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/divide_no_nan>
op_divide_no_nan <-
function (x1, x2)
keras$ops$divide_no_nan(x1, x2)
