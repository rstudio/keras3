#' Return a tensor of ones with the same shape and type of `x`.
#'
#' @description
#'
#' # Returns
#'     A tensor of ones with the same shape and type as `x`.
#'
#' @param x Input tensor.
#' @param dtype Overrides the data type of the result.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/ones_like>
k_ones_like <-
function (x, dtype = NULL)
keras$ops$ones_like(x, dtype)
