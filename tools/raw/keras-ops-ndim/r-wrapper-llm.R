#' Return the number of dimensions of a tensor.
#'
#' @description
#'
#' # Returns
#'     The number of dimensions in `x`.
#'
#' @param x Input tensor.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/ndim>
k_ndim <-
function (x)
keras$ops$ndim(x)
