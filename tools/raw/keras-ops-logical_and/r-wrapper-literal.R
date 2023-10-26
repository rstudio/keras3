#' Computes the element-wise logical AND of the given input tensors.
#'
#' @description
#' Zeros are treated as `False` and non-zeros are treated as `True`.
#'
#' # Returns
#'     Output tensor, element-wise logical AND of the inputs.
#'
#' @param x1 Input tensor.
#' @param x2 Input tensor.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/logical_and>
k_logical_and <-
function (x1, x2)
keras$ops$logical_and(x1, x2)
