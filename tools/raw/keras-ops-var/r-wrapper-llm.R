#' Compute the variance along the specified axes.
#'
#' @description
#'
#' # Returns
#'     Output tensor containing the variance.
#'
#' @param x Input tensor.
#' @param axis Axis or axes along which the variance is computed. The default
#'     is to compute the variance of the flattened tensor.
#' @param keepdims If this is set to `True`, the axes which are reduced are left
#'     in the result as dimensions with size one.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/var>
k_var <-
function (x, axis = NULL, keepdims = FALSE)
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$ops$var, args)
}
