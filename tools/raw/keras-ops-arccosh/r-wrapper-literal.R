#' Inverse hyperbolic cosine, element-wise.
#'
#' @description
#'
#' # Returns
#' Output tensor of same shape as x.
#'
#' # Examples
#' ```python
#' x = keras.ops.convert_to_tensor([10, 100])
#' keras.ops.arccosh(x)
#' # array([2.993223, 5.298292], dtype=float32)
#' ```
#'
#' @param x Input tensor.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/arccosh>
k_arccosh <-
function (x)
keras$ops$arccosh(x)
