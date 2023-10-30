#' Returns the indices of the bins to which each value in `x` belongs.
#'
#' @description
#' Output array of indices, of same shape as `x`.
#'
#' @param x Input array to be binned.
#' @param bins Array of bins. It has to be one-dimensional and monotonically
#'     increasing.
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/digitize>
#'
#' @examples
#' x <- c(0.0, 1.0, 3.0, 1.6)
#' bins <- c(0.0, 3.0, 4.5, 7.0)
#' keras::digitize(x, bins)
#' # array([1, 1, 2, 1])
k_digitize <-
function (x, bins)
keras$ops$digitize(x, bins)
