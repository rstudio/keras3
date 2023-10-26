#' Returns the cumulative sum of elements along a given axis.
#'
#' @description
#'
#' # Returns
#'     Output tensor.
#'
#' @param x Input tensor.
#' @param axis Axis along which the cumulative sum is computed.
#'     By default the input is flattened.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/cumsum>
k_cumsum <-
function (x, axis = NULL)
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$ops$cumsum, args)
}
