#' Join a sequence of tensors along a new axis.
#'
#' @description
#' The `axis` parameter specifies the index of the new axis in the
#' dimensions of the result.
#'
#' # Returns
#'     The stacked tensor.
#'
#' @param x A sequence of tensors.
#' @param axis Axis along which to stack. Defaults to `0`.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/stack>
k_stack <-
function (x, axis = 0L)
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$ops$stack, args)
}
