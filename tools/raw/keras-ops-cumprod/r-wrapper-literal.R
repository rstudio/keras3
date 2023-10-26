#' Return the cumulative product of elements along a given axis.
#'
#' @description
#'
#' # Returns
#'     Output tensor.
#'
#' @param x Input tensor.
#' @param axis Axis along which the cumulative product is computed.
#'     By default the input is flattened.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/cumprod>
k_cumprod <-
function (x, axis = NULL)
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$ops$cumprod, args)
}
