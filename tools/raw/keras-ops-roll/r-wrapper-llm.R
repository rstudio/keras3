#' Roll tensor elements along a given axis.
#'
#' @description
#' Elements that roll beyond the last position are re-introduced at the first.
#'
#' # Returns
#'     Output tensor.
#'
#' @param x Input tensor.
#' @param shift The number of places by which elements are shifted.
#' @param axis The axis along which elements are shifted. By default, the
#'     array is flattened before shifting, after which the original
#'     shape is restored.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/roll>
k_roll <-
function (x, shift, axis = NULL)
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$ops$roll, args)
}
