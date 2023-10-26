#' Return the number of elements in a tensor.
#'
#' @description
#'
#' # Returns
#'     Number of elements in `x`.
#'
#' @param x Input tensor.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/size>
k_size <-
function (x)
keras$ops$size(x)
