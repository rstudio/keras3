#' Compute the absolute value element-wise.
#'
#' @description
#' `keras.ops.abs` is a shorthand for this function.
#'
#' # Returns
#' An array containing the absolute value of each element in `x`.
#'
#' # Examples
#' ```python
#' x = keras.ops.convert_to_tensor([-1.2, 1.2])
#' keras.ops.absolute(x)
#' # array([1.2, 1.2], dtype=float32)
#' ```
#'
#' @param x Input tensor.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/absolute>
k_absolute <-
function (x)
keras$ops$absolute(x)
