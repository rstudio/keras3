#' Computes the inverse error function of `x`, element-wise.
#'
#' @description
#'
#' # Examples
#' ```python
#' x = np.array([-0.5, -0.2, -0.1, 0.0, 0.3])
#' keras.ops.erfinv(x)
#' # array([-0.47694, -0.17914, -0.08886,  0. ,  0.27246], dtype=float32)
#' ```
#'
#' @returns
#' A tensor with the same dtype as `x`.
#'
#' @param x
#' Input tensor.
#'
#' @export
#' @tether keras.ops.erfinv
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/erfinv>
op_erfinv <-
function (x)
keras$ops$erfinv(x)