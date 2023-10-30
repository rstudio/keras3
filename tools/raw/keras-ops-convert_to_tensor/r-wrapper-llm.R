#' Convert a R array to a tensor.
#'
#' @description
#'
#' # Returns
#' A tensor of the specified `dtype`.
#'
#' @param x A R array.
#' @param dtype The target type.
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/convert_to_tensor>
#'
#' @examples
#' x <- c(1, 2, 3)
#' y <- keras::k_convert_to_tensor(x)
k_convert_to_tensor <-
function (x, dtype = NULL)
keras$ops$convert_to_tensor(x, dtype)
