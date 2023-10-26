#' Expand the shape of a tensor.
#'
#' @description
#' Insert a new axis at the `axis` position in the expanded tensor shape.
#'
#' # Returns
#'     Output tensor with the number of dimensions increased.
#'
#' @param x Input tensor.
#' @param axis Position in the expanded axes where the new axis
#'     (or axes) is placed.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/expand_dims>
k_expand_dims <-
function (x, axis)
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$ops$expand_dims, args)
}
