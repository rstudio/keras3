#' Log-softmax activation function.
#'
#' @description
#' It is defined as:
#' `f(x) = x - max(x) - log(sum(exp(x - max(x))))`
#'
#' # Returns
#' A tensor with the same shape as `x`.
#'
#' # Examples
#' ```python
#' x = np.array([-1., 0., 1.])
#' x_log_softmax = keras.ops.log_softmax(x)
#' print(x_log_softmax)
#' # array([-2.40760596, -1.40760596, -0.40760596], shape=(3,), dtype=float64)
#' ```
#'
#' @param x Input tensor.
#' @param axis Integer, axis along which the log-softmax is applied.
#'     Defaults to `-1`.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/log_softmax>
k_log_softmax <-
function (x, axis = -1L)
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$ops$log_softmax, args)
}
