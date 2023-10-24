#' Return `(x1 != x2)` element-wise.
#'
#' @description
#'
#' # Returns
#'     Output tensor, element-wise comparsion of `x1` and `x2`.
#'
#' @param x1 First input tensor.
#' @param x2 Second input tensor.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/not_equal>
k_not_equal <-
function (x1, x2)
keras$ops$not_equal(x1, x2)
