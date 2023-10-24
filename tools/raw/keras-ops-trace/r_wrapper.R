#' Return the sum along diagonals of the tensor.
#'
#' @description
#' If `x` is 2-D, the sum along its diagonal with the given offset is
#' returned, i.e., the sum of elements `x[i, i+offset]` for all `i`.
#'
#' If a has more than two dimensions, then the axes specified by `axis1`
#' and `axis2` are used to determine the 2-D sub-arrays whose traces are
#' returned.
#'
#' The shape of the resulting tensor is the same as that of `x` with `axis1`
#' and `axis2` removed.
#'
#' # Returns
#' If `x` is 2-D, the sum of the diagonal is returned. If `x` has
#' larger dimensions, then a tensor of sums along diagonals is
#' returned.
#'
#' @param x Input tensor.
#' @param offset Offset of the diagonal from the main diagonal. Can be
#'     both positive and negative. Defaults to `0`.
#' @param axis1 Axis to be used as the first axis of the 2-D sub-arrays.
#'     Defaults to `0`.(first axis).
#' @param axis2 Axis to be used as the second axis of the 2-D sub-arrays.
#'     Defaults to `1` (second axis).
#'
#' @export
#' @family ops
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/trace>
k_trace <-
function (x, offset = 0L, axis1 = 0L, axis2 = 1L)
{
    args <- capture_args2(list(offset = as_integer, axis1 = as_integer,
        axis2 = as_integer))
    do.call(keras$ops$trace, args)
}
