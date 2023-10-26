#' Return upper triangle of a tensor.
#'
#' @description
#' For tensors with `ndim` exceeding 2, `triu` will apply to the
#' final two axes.
#'
#' # Returns
#'     Upper triangle of `x`, of same shape and data type as `x`.
#'
#' @param x Input tensor.
#' @param k Diagonal below which to zero elements. Defaults to `0`. the
#'     main diagonal. `k < 0` is below it, and `k > 0` is above it.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/triu>
k_triu <-
function (x, k = 0L)
{
    args <- capture_args2(list(k = as_integer))
    do.call(keras$ops$triu, args)
}
