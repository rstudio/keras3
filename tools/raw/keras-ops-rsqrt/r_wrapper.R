#' Computes reciprocal of square root of x element-wise.
#'
#' @description
#'
#' # Returns
#' A tensor with the same dtype as `x`.
#'
#' # Examples
#' ```python
#' x = keras.ops.convert_to_tensor([1.0, 10.0, 100.0])
#' keras.ops.rsqrt(x)
#' # array([1.0, 0.31622776, 0.1], dtype=float32)
#' ```
#'
#' @param x input tensor
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/rsqrt>
k_rsqrt <-
function (x)
keras$ops$rsqrt(x)
