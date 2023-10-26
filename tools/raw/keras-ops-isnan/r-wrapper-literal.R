#' Test element-wise for NaN and return result as a boolean tensor.
#'
#' @description
#'
#' # Returns
#'     Output boolean tensor.
#'
#' @param x Input tensor.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/isnan>
k_isnan <-
function (x)
keras$ops$isnan(x)
