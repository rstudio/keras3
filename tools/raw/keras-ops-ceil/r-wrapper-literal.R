#' Return the ceiling of the input, element-wise.
#'
#' @description
#' The ceil of the scalar `x` is the smallest integer `i`, such that
#' `i >= x`.
#'
#' # Returns
#'     The ceiling of each element in `x`, with float dtype.
#'
#' @param x Input tensor.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/ceil>
k_ceil <-
function (x)
keras$ops$ceil(x)
