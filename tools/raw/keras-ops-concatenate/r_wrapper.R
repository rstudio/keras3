#' Join a sequence of tensors along an existing axis.
#'
#' @description
#'
#' # Returns
#'     The concatenated tensor.
#'
#' @param xs The sequence of tensors to concatenate.
#' @param axis The axis along which the tensors will be joined. Defaults to `0`.
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/concatenate>
k_concatenate <-
function (xs, axis = 0L)
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$ops$concatenate, args)
}
