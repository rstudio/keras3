#' Inverse hyperbolic tangent, element-wise.
#'
#' @description
#'
#' # Returns
#'     Output tensor of same shape as `x`.
#'
#' @param x Input tensor.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/arctanh>
k_arctanh <-
function (x)
keras$ops$arctanh(x)
