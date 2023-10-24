#' Returns the indices of the bins to which each value in `x` belongs.
#'
#' @description
#'
#' # Returns
#' Output array of indices, of same shape as `x`.
#'
#' # Examples
#' ```python
#' x = np.array([0.0, 1.0, 3.0, 1.6])
#' bins = np.array([0.0, 3.0, 4.5, 7.0])
#' keras.ops.digitize(x, bins)
#' # array([1, 1, 2, 1])
#' ```
#'
#' @param x Input array to be binned.
#' @param bins Array of bins. It has to be one-dimensional and monotonically
#'     increasing.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/digitize>
k_digitize <-
function (x, bins)
keras$ops$digitize(x, bins)
