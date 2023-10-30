#' Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.
#'
#' @description
#' Returns a list of tensors unpacked along the given axis.
#'
#' @param x The input tensor.
#' @param num The length of the dimension axis. Automatically inferred
#'     if `NULL`.
#' @param axis The axis along which to unpack.
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/unstack>
#'
#' @examples
#' x <- array(c(1, 2, 3, 4), dim = c(2, 2))
#' unstack(x, axis = 1)
#' # list(array(c(1, 2)), array(c(3, 4)))
k_unstack <-
function (x, num = NULL, axis = 0L)
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$ops$unstack, args)
}
