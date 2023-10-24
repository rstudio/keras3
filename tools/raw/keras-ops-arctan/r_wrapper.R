#' Trigonometric inverse tangent, element-wise.
#'
#' @description
#'
#' # Returns
#' Tensor of the inverse tangent of each element in `x`, in the interval
#' `[-pi/2, pi/2]`.
#'
#' # Examples
#' ```python
#' x = keras.ops.convert_to_tensor([0, 1])
#' keras.ops.arctan(x)
#' # array([0., 0.7853982], dtype=float32)
#' ```
#'
#' @param x Input tensor.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/arctan>
k_arctan <-
function (x)
keras$ops$arctan(x)
