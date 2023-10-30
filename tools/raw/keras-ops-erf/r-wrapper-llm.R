#' Computes the error function of `x`, element-wise.
#'
#' @description
#' Returns a tensor with the same dtype as `x`.
#'
#' @param x Input tensor.
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/erf>
#'
#' @examples
#' x <- c(-3.0, -2.0, -1.0, 0.0, 1.0)
#' keras::k_erf(x)
#' # array([-0.99998 , -0.99532, -0.842701,  0.,  0.842701], dtype=float32)
k_erf <-
function (x)
keras$ops$erf(x)
