#' Return the element-wise square of the input.
#'
#' @description
#'
#' # Returns
#'     Output tensor, the square of `x`.
#'
#' @param x Input tensor.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/square>
k_square <-
function (x)
keras$ops$square(x)
