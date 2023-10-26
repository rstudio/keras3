#' Replace NaN with zero and infinity with large finite numbers.
#'
#' @description
#'
#' # Returns
#'     `x`, with non-finite values replaced.
#'
#' @param x Input data.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/nan_to_num>
k_nan_to_num <-
function (x)
keras$ops$nan_to_num(x)
