#' Inverse hyperbolic sine, element-wise.
#'
#' @description
#' Returns output tensor of same shape as `x`.
#'
#' @param x Input tensor.
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/arcsinh>
#'
#' @examples
#' x <- tf$convert_to_tensor(c(1, -1, 0))
#' keras::k_arcsinh(x)
#' # array([0.88137364, -0.88137364, 0.0], dtype=float32)
k_arcsinh <-
function (x)
keras$ops$arcsinh(x)
