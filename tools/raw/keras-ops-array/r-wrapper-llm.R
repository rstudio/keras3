#' Create a tensor.
#'
#' @description
#' Returns a tensor.
#'
#' @param x Input tensor.
#' @param dtype The desired data-type for the tensor.
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/array>
#'
#' @examples
#' keras::k_array(c(1, 2, 3))
#' # array([1, 2, 3], dtype=int32)
#'
#' keras::k_array(c(1, 2, 3), dtype="float32")
#' # array([1., 2., 3.], dtype=float32)
k_array <-
function (x, dtype = NULL)
keras$ops$array(x, dtype)
