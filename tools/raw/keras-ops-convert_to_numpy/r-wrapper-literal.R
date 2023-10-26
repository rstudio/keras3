#' Convert a tensor to a NumPy array.
#'
#' @description
#'
#' # Returns
#'     A NumPy array.
#'
#' @param x A tensor.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/convert_to_numpy>
k_convert_to_numpy <-
function (x)
keras$ops$convert_to_numpy(x)
