#' Sigmoid activation function.
#'
#' @description
#' It is defined as `f(x) = 1 / (1 + exp(-x))`.
#'
#' @param x Input tensor.
#' @return A tensor with the same shape as `x`.
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/sigmoid>
#'
#' @examples
#' x <- tf$convert_to_tensor(c(-6.0, 1.0, 0.0, 1.0, 6.0))
#' keras::k_sigmoid(x)
#' # array([0.00247262, 0.7310586, 0.5, 0.7310586, 0.9975274], dtype=float32)
k_sigmoid <-
function (x)
keras$ops$sigmoid(x)
