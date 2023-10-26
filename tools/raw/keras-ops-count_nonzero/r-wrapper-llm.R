#' Counts the number of non-zero values in `x` along the given `axis`.
#'
#' @description
#' If no axis is specified then all non-zeros in the tensor are counted.
#'
#' # Returns
#' int or tensor of ints.
#'
#' # Examples
#' ```python
#' x = keras.ops.array([[0, 1, 7, 0], [3, 0, 2, 19]])
#' keras.ops.count_nonzero(x)
#' # 5
#' keras.ops.count_nonzero(x, axis=0)
#' # array([1, 1, 2, 1], dtype=int64)
#' keras.ops.count_nonzero(x, axis=1)
#' # array([2, 3], dtype=int64)
#' ```
#'
#' @param x Input tensor.
#' @param axis Axis or tuple of axes along which to count the number of
#'     non-zeros. Defaults to `None`.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/count_nonzero>
k_count_nonzero <-
function (x, axis = NULL)
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$ops$count_nonzero, args)
}
