#' Broadcast a tensor to a new shape.
#'
#' @description
#' Returns a tensor with the desired shape.
#'
#' @param x The tensor to broadcast.
#' @param shape The shape of the desired tensor. A single integer `i` is
#'     interpreted as `(i,)`.
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/broadcast_to>
#'
#' @examples
#' x <- tf$constant(c(1, 2, 3))
#' tf$broadcast_to(x, shape(3, 3))
#' # array([[1, 2, 3],
#' #        [1, 2, 3],
#' #        [1, 2, 3]])
k_broadcast_to <-
function (x, shape)
{
    args <- capture_args2(list(shape = as_integer))
    do.call(keras$ops$broadcast_to, args)
}
