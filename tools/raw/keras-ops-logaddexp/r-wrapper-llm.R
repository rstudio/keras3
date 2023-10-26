#' Logarithm of the sum of exponentiations of the inputs.
#'
#' @description
#' Calculates `log(exp(x1) + exp(x2))`.
#'
#' # Returns
#' Output tensor, element-wise logarithm of the sum of exponentiations
#' of the inputs.
#'
#' @param x1 Input tensor.
#' @param x2 Input tensor.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/logaddexp>
k_logaddexp <-
function (x1, x2)
keras$ops$logaddexp(x1, x2)
