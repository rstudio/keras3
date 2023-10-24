#' Count the number of occurrences of each value in a tensor of integers.
#'
#' @description
#'
#' # Returns
#' 1D tensor where each element gives the number of occurrence(s) of its
#' index value in x. Its length is the maximum between `max(x) + 1` and
#' minlength.
#'
#' # Examples
#' ```python
#' x = keras.ops.array([1, 2, 2, 3], dtype="uint8")
#' keras.ops.bincount(x)
#' # array([0, 1, 2, 1], dtype=int32)
#' weights = x / 2
#' weights
#' # array([0.5, 1., 1., 1.5], dtype=float64)
#' keras.ops.bincount(x, weights=weights)
#' # array([0., 0.5, 2., 1.5], dtype=float64)
#' minlength = (keras.ops.max(x).numpy() + 1) + 2 # 6
#' keras.ops.bincount(x, minlength=minlength)
#' # array([0, 1, 2, 1, 0, 0], dtype=int32)
#' ```
#'
#' @param x Input tensor.
#'     It must be of dimension 1, and it must only contain non-negative
#'     integer(s).
#' @param weights Weight tensor.
#'     It must have the same length as `x`. The default value is `None`.
#'     If specified, `x` is weighted by it, i.e. if `n = x[i]`,
#'     `out[n] += weight[i]` instead of the default behavior `out[n] += 1`.
#' @param minlength An integer.
#'     The default value is 0. If specified, there will be at least
#'     this number of bins in the output tensor. If greater than
#'     `max(x) + 1`, each value of the output at an index higher than
#'     `max(x)` is set to 0.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/bincount>
k_bincount <-
function (x, weights = NULL, minlength = 0L)
{
    args <- capture_args2(list(x = as_integer, minlength = as_integer))
    do.call(keras$ops$bincount, args)
}
