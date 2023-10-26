#' Interchange two axes of a tensor.
#'
#' @description
#'
#' # Returns
#'     A tensor with the axes swapped.
#'
#' @param x Input tensor.
#' @param axis1 First axis.
#' @param axis2 Second axis.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/swapaxes>
k_swapaxes <-
function (x, axis1, axis2)
keras$ops$swapaxes(x, axis1, axis2)
