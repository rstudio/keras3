#' Scaled Exponential Linear Unit (SELU) activation function.
#'
#' @description
#' It is defined as:
#'
#' `f(x) =  scale * alpha * (exp(x) - 1.) for x < 0`,
#' `f(x) = scale * x for x >= 0`.
#'
#' # Returns
#' A tensor with the same shape as `x`.
#'
#' # Examples
#' ```python
#' x = np.array([-1., 0., 1.])
#' x_selu = keras.ops.selu(x)
#' print(x_selu)
#' # array([-1.11133055, 0., 1.05070098], shape=(3,), dtype=float64)
#' ```
#'
#' @param x Input tensor.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/selu>
k_selu <-
function (x)
keras$ops$selu(x)
