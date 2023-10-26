#' Take elements from a tensor along an axis.
#'
#' @description
#'
#' # Returns
#'     The corresponding tensor of values.
#'
#' @param x Source tensor.
#' @param indices The indices of the values to extract.
#' @param axis The axis over which to select values. By default, the
#'     flattened input tensor is used.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/take>
k_take <-
function (x, indices, axis = NULL)
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$ops$take, args)
}
