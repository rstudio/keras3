#' Return a tensor of zeros with the same shape and type as `x`.
#'
#' @description
#'
#' # Returns
#'     A tensor of zeros with the same shape and type as `x`.
#'
#' @param x Input tensor.
#' @param dtype Overrides the data type of the result.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/zeros_like>
k_zeros_like <-
function (x, dtype = NULL)
keras$ops$zeros_like(x, dtype)
