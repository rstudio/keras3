#' Computes the error function of `x`, element-wise.
#'
#' @description
#'
#' # Returns
#' A tensor with the same dtype as `x`.
#'
#' # Examples
#' ```python
#' x = np.array([-3.0, -2.0, -1.0, 0.0, 1.0])
#' keras.ops.erf(x)
#' # array([-0.99998 , -0.99532, -0.842701,  0.,  0.842701], dtype=float32)
#' ```
#'
#' @param x Input tensor.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/erf>
k_erf <-
function (x)
keras$ops$erf(x)
