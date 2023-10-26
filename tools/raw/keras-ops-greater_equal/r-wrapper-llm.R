#' Return the truth value of `x1 >= x2` element-wise.
#'
#' @description
#'
#' # Returns
#'     Output tensor, element-wise comparison of `x1` and `x2`.
#'
#' @param x1 First input tensor.
#' @param x2 Second input tensor.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/greater_equal>
k_greater_equal <-
function (x1, x2)
keras$ops$greater_equal(x1, x2)
