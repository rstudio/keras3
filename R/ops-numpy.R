#' Safe element-wise division which returns 0 where the denominator is 0.
#'
#' @returns
#' The quotient `x1/x2`, element-wise, with zero where x2 is zero.
#'
#' @param x1
#' First input tensor.
#'
#' @param x2
#' Second input tensor.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.divide_no_nan
# @seealso
# + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/divide_no_nan>
op_divide_no_nan <-
function (x1, x2)
keras$ops$divide_no_nan(x1, x2)


#' Performs an indirect partition along the given axis.
#'
#' @description
#' It returns an array
#' of indices of the same shape as `x` that index data along the given axis
#' in partitioned order.
#'
#' @returns
#' Array of indices that partition `x` along the specified `axis`.
#'
#' @param x
#' Array to sort.
#'
#' @param kth
#' Element index to partition by.
#' The k-th element will be in its final sorted position and all
#' smaller elements will be moved before it and all larger elements
#' behind it. The order of all elements in the partitions is undefined.
#' If provided with a sequence of k-th it will partition all of them
#' into their sorted position at once.
#'
#' @param axis
#' Axis along which to sort. The default is -1 (the last axis).
#' If `NULL`, the flattened array is used.
#'
#' @export
#' @family numpy ops
#' @family ops
#' @tether keras.ops.argpartition
op_argpartition <-
function (x, kth, axis = -1L)
{
    args <- capture_args(list(axis = as_axis))
    do.call(keras$ops$argpartition, args)
}
