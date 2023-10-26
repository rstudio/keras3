#' Rectified linear unit activation function.
#'
#' @description
#' It is defined as `f(x) = max(0, x)`.
#'
#' # Returns
#' A tensor with the same shape as `x`.
#'
#' # Examples
#' ```python
#' x1 = keras.ops.convert_to_tensor([-1.0, 0.0, 1.0, 0.2])
#' keras.ops.relu(x1)
#' # array([0.0, 0.0, 1.0, 0.2], dtype=float32)
#' ```
#'
#' @param x Input tensor.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/relu>
k_relu <-
function (x)
keras$ops$relu(x)
