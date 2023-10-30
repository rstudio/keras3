#' Softplus activation function.
#'
#' @description
#' It is defined as `f(x) = log(exp(x) + 1)`, where `log` is the natural
#' logarithm and `exp` is the exponential function.
#'
#' # Returns
#' A tensor with the same shape as `x`.
#'
#' @param x Input tensor.
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/softplus>
#'
#' @examples
#' x <- tf$convert_to_tensor(c(-0.555, 0.0, 0.555))
#' keras::k_softplus(x)
#' # array([0.45366603, 0.6931472, 1.008666], dtype=float32)
k_softplus <-
function (x)
keras$ops$softplus(x)
