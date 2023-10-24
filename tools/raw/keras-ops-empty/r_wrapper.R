#' Return a tensor of given shape and type filled with uninitialized data.
#'
#' @description
#'
#' # Returns
#'     The empty tensor.
#'
#' @param shape Shape of the empty tensor.
#' @param dtype Desired data type of the empty tensor.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/empty>
k_empty <-
function (shape, dtype = NULL)
keras$ops$empty(shape, dtype)
