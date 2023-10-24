#' Returns `(x1 == x2)` element-wise.
#'
#' @description
#'
#' # Returns
#'     Output tensor, element-wise comparison of `x1` and `x2`.
#'
#' @param x1 Tensor to compare.
#' @param x2 Tensor to compare.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/equal>
k_equal <-
function (x1, x2)
keras$ops$equal(x1, x2)
