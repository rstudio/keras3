#' Numerical negative, element-wise.
#'
#' @description
#'
#' # Returns
#'     Output tensor, `y = -x`.
#'
#' @param x Input tensor.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/negative>
k_negative <-
function (x)
keras$ops$negative(x)
